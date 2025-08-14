from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json

load_dotenv()
base_url = os.getenv("BASE_URL", "http://localhost:1234/v1")
api_key = os.getenv("API_KEY", "lm-studio")

class AnswerWithJustification(BaseModel):
    """An answer to the user's question along with justification for the answer."""

    answer: str
    """The answer to the user's question"""
    justification: str
    """Justification for the answer"""

llm = ChatOpenAI(base_url=base_url, api_key=api_key, temperature=0)

# Alternative approach: Use prompt engineering to get structured output
prompt = """Answer the following question and provide your response in JSON format with the following structure:
{
    "answer": "your answer here",
    "justification": "your justification here"
}

Question: What weighs more, a pound of bricks or a pound of feathers?

Response (JSON only):"""

response = llm.invoke(prompt)
print("Raw response:", response.content)

# Try to parse the JSON response
try:
    json_response = json.loads(response.content)
    structured_response = AnswerWithJustification(**json_response)
    print("Structured response:", structured_response)
except Exception as e:
    print(f"Could not parse as structured output: {e}")
    print("Content:", response.content)
