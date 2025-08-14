from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import time
import os
from typing import Any

load_dotenv()
base_url: str = os.getenv("BASE_URL", "http://localhost:1234/v1")
api_key: str = os.getenv("API_KEY", "lm-studio")

template: PromptTemplate = PromptTemplate.from_template("""Answer the question based on the context below. If the question cannot be answered using the information provided, answer with \"I don't know\".

Context: {context}

Question: {question}

Answer: """)

response: Any = template.invoke(
    {
        "context": "The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.",
        "question": "Which model providers offer LLMs?",
    }
)

print(response)
