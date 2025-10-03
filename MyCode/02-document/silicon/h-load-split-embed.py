from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
base_url = os.environ.get("BASE_URL")
api_key = os.environ.get("API_KEY")
model_name = os.environ.get("MODEL")

embeddings_model = OpenAIEmbeddings(model=model_name, base_url=base_url, api_key=api_key)
# Load the document
loader = TextLoader("./test.txt", encoding="utf-8")
doc = loader.load()

# Split the document
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(doc)

# Generate embeddings in batches
BATCH_SIZE = 64
all_embeddings = []
chunk_texts = [chunk.page_content for chunk in chunks]
for i in range(0, len(chunk_texts), BATCH_SIZE):
    batch = chunk_texts[i:i+BATCH_SIZE]
    batch_embeddings = embeddings_model.embed_documents(batch)
    all_embeddings.extend(batch_embeddings)

print(all_embeddings)
