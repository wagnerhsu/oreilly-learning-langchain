from common import create_chat_model
from langchain_core.messages import HumanMessage

model = create_chat_model()
prompt = [HumanMessage("What is the capital of France?")]

response = model.invoke(prompt)
print(response.content)
