from math import ceil

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

import os
from dotenv import load_dotenv
load_dotenv()
base_url = os.environ.get("BASE_URL")
api_key = os.environ.get("API_KEY")
model_name = os.environ.get("MODEL")
embedding_model_name = os.environ.get("EMBEDDING_MODEL")

# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

# Load the document, split it into chunks
raw_documents = TextLoader('./test.txt', encoding='utf-8').load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# Create embeddings for the documents
embeddings_model = OpenAIEmbeddings(model=embedding_model_name,base_url=base_url, api_key=api_key)

# Batch documents to avoid exceeding API batch size limit
BATCH_SIZE = 64
num_batches = ceil(len(documents) / BATCH_SIZE)
db = None
for i in range(num_batches):
    batch_docs = documents[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    if i == 0:
        db = PGVector.from_documents(batch_docs, embeddings_model, connection=connection)
    else:
        db.add_documents(batch_docs)

# create retriever to retrieve 2 relevant documents
retriever = db.as_retriever(search_kwargs={"k": 5})

prompt_rag_fusion = ChatPromptTemplate.from_template(
    """You are a helpful assistant that generates multiple search queries based on a single input query. \n Generate multiple search queries related to: {question} \n Output (4 queries):""")


def parse_queries_output(message):
    # Filter out any empty strings that may result from splitting
    return [q.strip() for q in message.content.split('\n') if q.strip()]


llm = ChatOpenAI(model=model_name,base_url=base_url,api_key=api_key, temperature=0)
query_gen = prompt_rag_fusion | llm | parse_queries_output

query = "Who are the key figures in the ancient greek history of philosophy?"

generated_queries = query_gen.invoke(query)

print("generated queries: ", generated_queries)


"""
we fetch relevant documents for each query and pass them into a function to rerank (that is, reorder according to relevancy) the final list of relevant documents.
"""


def reciprocal_rank_fusion(results: list[list], k=60):
    """reciprocal rank fusion on multiple lists of ranked documents and an optional parameter k used in the RRF formula"""
    # Initialize a dictionary to hold fused scores for each document
    # Documents will be keyed by their contents to ensure uniqueness
    fused_scores = {}
    documents = {}
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            doc_str = doc.page_content
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc
            fused_scores[doc_str] += 1 / (rank + k)
    # sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_doc_strs = sorted(
        fused_scores, key=lambda d: fused_scores[d], reverse=True)
    return [documents[doc_str] for doc_str in reranked_doc_strs]


retrieval_chain = query_gen | (lambda queries: retriever.batch(queries)) | reciprocal_rank_fusion

result = retrieval_chain.invoke(query)

print("retrieved context using rank fusion: ", result[0].page_content)
print("\n\n")

print("Use model to answer question based on retrieved docs\n")


prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)

query = "Who are the some important yet not well known philosophers in the ancient greek history of philosophy?"


@chain
def rag_fusion(input):
    # fetch relevant documents
    docs = retrieval_chain.invoke(input)  # format prompt
    formatted = prompt.invoke(
        {"context": docs, "question": input})  # generate answer
    answer = llm.invoke(formatted)
    return answer


# run
print("Running rag fusion\n")
result = rag_fusion.invoke(query)
print(result.content)
