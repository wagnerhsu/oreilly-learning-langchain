from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
base_url = os.getenv("BASE_URL", "http://localhost:1234/v1")
api_key = os.getenv("API_KEY", "lm-studio")

model = ChatOpenAI(model="gpt-3.5-turbo", base_url=base_url, api_key=api_key)

completion = model.invoke("Hi there!")
# Hi!

completions = model.batch(["Hi there!", "Bye!"])
# ['Hi!', 'See you!']

for token in model.stream("Bye!"):
    print(token)
    # Good
    # bye
    # !
