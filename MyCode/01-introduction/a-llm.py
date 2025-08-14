from langchain_openai.chat_models import ChatOpenAI
import time
from dotenv import load_dotenv
import os

load_dotenv()
base_url = os.getenv("BASE_URL", "http://localhost:1234/v1")
api_key = os.getenv("API_KEY", "lm-studio")

model = ChatOpenAI(base_url=base_url, api_key=api_key, model="gpt-3.5-turbo")

start_time = time.time()
response = model.invoke("The sky is")
end_time = time.time()
elapsed = end_time - start_time

print(response.content)

print(f"Response time: {elapsed:.2f} seconds")

# If your backend supports token usage, you may be able to access it like this:

if hasattr(response, "usage_metadata"):
    tokens_used = response.usage_metadata["total_tokens"]
    tokens_per_sec = tokens_used / elapsed if elapsed > 0 else 0
    print(f"Tokens used: {tokens_used}")
    print(f"Tokens per second: {tokens_per_sec:.2f}")
else:
    print("Token usage_metadata information is not available for this response.")
