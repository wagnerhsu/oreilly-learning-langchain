from langchain_core.runnables import chain
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
base_url = os.getenv("BASE_URL", "http://localhost:1234/v1")
api_key = os.getenv("API_KEY", "lm-studio")

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

model = ChatOpenAI(base_url=base_url, api_key=api_key, model="gpt-3.5-turbo")


@chain
async def chatbot(values):
    prompt = await template.ainvoke(values)
    return await model.ainvoke(prompt)


async def main():
    return await chatbot.ainvoke({"question": "Which model providers offer LLMs?"})

if __name__ == "__main__":
    import asyncio
    print(asyncio.run(main()))
