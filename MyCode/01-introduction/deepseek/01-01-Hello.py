from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(model="deepseek-chat")
messages = [
        ("human", "Hello, world!"),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)
