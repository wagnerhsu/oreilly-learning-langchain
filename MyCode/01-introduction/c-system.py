
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
import time
import os
from typing import List

load_dotenv()
base_url: str = os.getenv("BASE_URL", "http://localhost:1234/v1")
api_key: str = os.getenv("API_KEY", "lm-studio")

model: ChatOpenAI = ChatOpenAI(base_url=base_url, api_key=api_key)
system_msg: SystemMessage = SystemMessage(
    "You are a helpful assistant that responds to questions with three exclamation marks."
)
human_msg: HumanMessage = HumanMessage("What is the capital of France?")

response: HumanMessage = model.invoke([system_msg, human_msg])
print(response.content)
