from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
import os

load_dotenv()
base_url = os.environ.get("BASE_URL")
api_key = os.environ.get("API_KEY")
model_name = os.environ.get("MODEL")

model = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key, temperature=0)
prompt = [HumanMessage("What is the capital of France?")]

response = model.invoke(prompt)
print(response.content)
