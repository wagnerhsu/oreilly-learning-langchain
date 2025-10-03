import os
import requests

# Load .env if present (python-dotenv). Graceful if not installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "model": "THUDM/GLM-Z1-9B-0414",
    "thinking_budget": 4096,
    "top_p": 0.7,
    "messages": [
        {
            "content": "What opportunities and challenges will the Chinese large model industry face in 2025?",
            "role": "user"
        }
    ]
}

# Read token from environment (SILICON_API_KEY or OPENAI_API_KEY)
token = os.environ.get("SILICON_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not token:
    raise RuntimeError(
        "API token not found. Set SILICON_API_KEY or OPENAI_API_KEY in your environment or in a .env file. "
        "See .env.example in this folder for format."
    )

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json())