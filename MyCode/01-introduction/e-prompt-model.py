from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import time
import os
from typing import Any
from common import create_chat_model


# both `template` and `model` can be reused many times

template: PromptTemplate = PromptTemplate.from_template(
    """Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: """
)

model: ChatOpenAI = create_chat_model(model_name="openai/gpt-oss-20b")

# `prompt` and `completion` are the results of using template and model once

prompt: Any = template.invoke(
    {
        "context": "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
        "question": "Which model providers offer LLMs?",
    }
)

response: Any = model.invoke(prompt)
print(response)
