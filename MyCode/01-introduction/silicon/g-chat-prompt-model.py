from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os
load_dotenv()
base_url = os.environ.get("BASE_URL")
api_key = os.environ.get("API_KEY")
model_name = os.environ.get("MODEL")

model = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key, temperature=0)
# both `template` and `model` can be reused many times

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            'Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don\'t know".',
        ),
        ("human", "Context: {context}"),
        ("human", "Question: {question}"),
    ]
)



# `prompt` and `completion` are the results of using template and model once

prompt = template.invoke(
    {
        "context": "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
        "question": "Which model providers offer LLMs?",
    }
)

print(model.invoke(prompt))
