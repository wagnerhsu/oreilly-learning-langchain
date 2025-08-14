from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()
base_url = os.getenv("BASE_URL", "http://localhost:1234/v1")
api_key = os.getenv("API_KEY", "lm-studio")

model = ChatOpenAI(base_url=base_url, api_key=api_key)
prompt = [HumanMessage("What is the capital of France?")]

response = model.invoke(prompt)
print(response.content)
