from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from dotenv import load_dotenv
import os
load_dotenv()
base_url = os.environ.get("BASE_URL")
api_key = os.environ.get("API_KEY")
model_name = os.environ.get("MODEL")

llm = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key, temperature=0)
class AnswerWithJustification(BaseModel):
    """An answer to the user's question along with justification for the answer."""

    answer: str
    """The answer to the user's question"""
    justification: str
    """Justification for the answer"""


structured_llm = llm.with_structured_output(AnswerWithJustification)

response = structured_llm.invoke(
    "What weighs more, a pound of bricks or a pound of feathers")
print(response)
