from common import create_chat_model
from langchain_core.messages import HumanMessage

model = create_chat_model(model_name="open/gpt-oss-20b")
prompt = [HumanMessage("What is the capital of France?")]

response = model.invoke(prompt)
print(response.content)
