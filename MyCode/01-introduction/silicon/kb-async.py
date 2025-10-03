from langchain_core.runnables import chain
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
base_url = os.environ.get("BASE_URL")
api_key = os.environ.get("API_KEY")
model_name = os.environ.get("MODEL")
model = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key, temperature=0)
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)



@chain
async def chatbot(values):
    prompt = await template.ainvoke(values)
    return await model.ainvoke(prompt)


async def main():
    return await chatbot.ainvoke({"question": "Which model providers offer LLMs?"})

if __name__ == "__main__":
    import asyncio
    print(asyncio.run(main()))
