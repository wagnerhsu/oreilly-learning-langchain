"""
Common utilities for LangChain projects.
This module contains frequently used functions and configurations.
"""

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr, BaseModel
from dotenv import load_dotenv
import os
import json
import time
import re
from typing import Any, Optional, Dict, Type


def load_environment() -> tuple[str, SecretStr]:
    """Load environment variables and return base_url and api_key."""
    load_dotenv()
    base_url: str = os.getenv("BASE_URL", "http://localhost:1234/v1")
    api_key_str: str = os.getenv("API_KEY", "lm-studio")
    api_key: SecretStr = SecretStr(api_key_str)
    return base_url, api_key

def clean_json_response(content: str) -> str:
    # 移除 ```json ... ``` 包裹
    content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
    content = re.sub(r'\s*```\s*$', '', content, flags=re.MULTILINE)
    # 移除首尾空白
    return content.strip()


def create_chat_model(model_name: str = "gpt-3.5-turbo", temperature: float = 0) -> ChatOpenAI:
    """Create and return a ChatOpenAI model with default configuration."""
    base_url, api_key = load_environment()
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=temperature
    )


def load_text_file(file_path: str, encoding: str = "utf-8") -> str:
    """Load and return the content of a text file."""
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def create_chat_prompt_from_files(
    system_message_file: str,
    question_message_file: str
) -> ChatPromptTemplate:
    """Create a ChatPromptTemplate from system and question message files."""
    system_message = load_text_file(system_message_file)
    question_message = load_text_file(question_message_file)

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "Context: {context}"),
        ("human", question_message),
    ])


def measure_response_time(func, *args, **kwargs) -> tuple[Any, float]:
    """Measure the execution time of a function and return result and elapsed time."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed = end_time - start_time
    return result, elapsed


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON object from text using regex."""
    json_match = re.search(r'{.*}', text, re.DOTALL)
    return json_match.group(0) if json_match else None


def parse_json_response(response_content: str, model_class: Optional[Type[BaseModel]] = None) -> tuple[bool, Any]:
    """
    Parse JSON response and optionally validate against a Pydantic model.
    Returns (success, result) tuple.
    """
    if not response_content or not response_content.strip():
        return False, "Response is empty"

    content_to_parse = response_content.strip()

    try:
        json_response = json.loads(content_to_parse)
        if model_class:
            structured_response = model_class(**json_response)
            return True, structured_response
        return True, json_response
    except json.JSONDecodeError as e:
        # Try to extract JSON from the response
        extracted_json = extract_json_from_text(content_to_parse)
        if extracted_json:
            try:
                json_response = json.loads(extracted_json)
                if model_class:
                    structured_response = model_class(**json_response)
                    return True, structured_response
                return True, json_response
            except Exception:
                return False, f"Could not parse extracted JSON: {e}"
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Other error: {e}"


def create_structured_prompt(question: str, response_schema: Dict[str, str]) -> str:
    """Create a prompt for structured JSON output."""
    schema_example = json.dumps(response_schema, indent=4)
    return f"""Answer the following question and provide your response in JSON format with the following structure:
{schema_example}

Question: {question}

Response (JSON only):"""


def print_debug_info(response: Any) -> None:
    """Print detailed debug information about a response."""
    print("Raw response:", response.content)
    print("Response type:", type(response.content))
    print("Response repr:", repr(response.content))
    print("Response length:", len(response.content) if response.content else 0)


# Common Pydantic models
class AnswerWithJustification(BaseModel):
    """An answer to the user's question along with justification for the answer."""
    answer: str
    justification: str


class SimpleAnswer(BaseModel):
    """A simple answer model."""
    answer: str


# Default prompts
DEFAULT_QA_SYSTEM_MESSAGE = """Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know"."""

DEFAULT_QUESTION_TEMPLATE = "Question: {question}"
