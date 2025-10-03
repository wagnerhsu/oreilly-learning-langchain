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

# Validate environment variables
missing_vars = []
if not base_url:
    missing_vars.append("BASE_URL")
if not api_key:
    missing_vars.append("API_KEY")
if not model_name:
    missing_vars.append("MODEL")
if not embedding_model_name:
    missing_vars.append("EMBEDDING_MODEL")
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Optionally, check for valid model names
valid_chat_models = ["THUDM/GLM-Z1-9B-0414"]
valid_embedding_models = ["BAAI/bge-m3"]
if model_name not in valid_chat_models:
    print(f"Warning: MODEL '{model_name}' may be invalid. Expected one of: {valid_chat_models}")
if embedding_model_name not in valid_embedding_models:
    print(f"Warning: EMBEDDING_MODEL '{embedding_model_name}' may be invalid. Expected one of: {valid_embedding_models}")

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

# instruction to generate multiple queries
perspectives_prompt = ChatPromptTemplate.from_template(
    """You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. 
    By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based  similarity search. 
    Provide these alternative questions separated by newlines. 
    Original question: {question}""")

llm = ChatOpenAI(model=model_name,base_url=base_url,api_key=api_key, temperature=0)


def parse_queries_output(message):
    # Filter out any empty strings that may result from splitting
    return [q.strip() for q in message.content.split('\n') if q.strip()]


query_gen = perspectives_prompt | llm | parse_queries_output


def get_unique_union(document_lists):
    # Flatten list of lists, and dedupe them
    deduped_docs = {
        doc.page_content: doc for sublist in document_lists for doc in sublist}
    # return a flat list of unique docs
    return list(deduped_docs.values())


# The retriever.batch method takes a list of queries and returns a list of document lists.
# We then pass this to get_unique_union to flatten and deduplicate.
retrieval_chain = query_gen | (lambda queries: retriever.batch(queries)) | get_unique_union

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)

query = "Who are the key figures in the ancient greek history of philosophy?"


@chain
def multi_query_qa(input):
    # fetch relevant documents
    docs = retrieval_chain.invoke(input)  # format prompt
    formatted = prompt.invoke(
        {"context": docs, "question": input})  # generate answer
    answer = llm.invoke(formatted)
    return answer


# run
print("Running multi query qa\n")
result = multi_query_qa.invoke(query)
print(result.content)
