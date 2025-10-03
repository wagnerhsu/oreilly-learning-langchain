from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
base_url = os.environ.get("BASE_URL")
api_key = os.environ.get("API_KEY")
model_name = os.environ.get("MODEL")

class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")


model = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key, temperature=0)
model = model.with_structured_output(Joke)

result = model.invoke("Tell me a joke about cats")
print(result)
