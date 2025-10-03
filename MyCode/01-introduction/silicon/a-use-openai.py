import os
from openai import OpenAI

# Load .env if present (python-dotenv). This is optional; if python-dotenv
# isn't installed the code will still fall back to environment variables.
try:
    from dotenv import load_dotenv
    # look for a .env file in the current working directory
    load_dotenv()
except Exception:
    # if dotenv isn't available, continue; we'll read from os.environ below
    pass

# Contract: read API key from environment variable SILICON_API_KEY or OPENAI_API_KEY
# Inputs: environment or .env file
# Output: configured OpenAI client
api_key = os.environ.get("SILICON_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "API key not found. Set SILICON_API_KEY or OPENAI_API_KEY in your environment or in a .env file. "
        "See .env.example for format."
    )

client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
response = client.chat.completions.create(
    # model='Pro/deepseek-ai/DeepSeek-R1',
    model="THUDM/GLM-Z1-9B-0414",
    messages=[
        {'role': 'user', 
        'content': "推理模型会给市场带来哪些新的机会"}
    ],
    stream=True
)

for chunk in response:
    if not chunk.choices:
        continue
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)