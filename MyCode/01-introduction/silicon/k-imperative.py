from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

from dotenv import load_dotenv
import os

load_dotenv()
base_url = os.environ.get("BASE_URL")
api_key = os.environ.get("API_KEY")
model_name = os.environ.get("MODEL")

model = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key, temperature=0)
# the building blocks

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)


# combine them in a function
# @chain decorator adds the same Runnable interface for any function you write


@chain
def chatbot(values):
    prompt = template.invoke(values)
    return model.invoke(prompt)


# use it

response = chatbot.invoke({"question": "Which model providers offer LLMs?"})
print(response.content)
