from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from dotenv import load_dotenv
import os

load_dotenv()
base_url = os.getenv("BASE_URL", "http://localhost:1234/v1")
api_key = os.getenv("API_KEY", "lm-studio")

# the building blocks

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

model = ChatOpenAI(model="gpt-3.5-turbo", base_url=base_url, api_key=api_key)

# combine them in a function
# @chain decorator adds the same Runnable interface for any function you write


@chain
def chatbot(values):
    prompt = template.invoke(values)
    return model.invoke(prompt)


# use it

response = chatbot.invoke({"question": "Which model providers offer LLMs?"})
print(response.content)
