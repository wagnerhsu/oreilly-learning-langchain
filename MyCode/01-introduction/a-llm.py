from common import create_chat_model, measure_response_time

model = create_chat_model(model_name="open/gpt-oss-20b")

response, elapsed = measure_response_time(model.invoke, "The sky is")

print(response.content)
print(f"Response time: {elapsed:.2f} seconds")

# If your backend supports token usage, you may be able to access it like this:

if hasattr(response, "usage_metadata"):
    tokens_used = response.usage_metadata["total_tokens"]
    tokens_per_sec = tokens_used / elapsed if elapsed > 0 else 0
    print(f"Tokens used: {tokens_used}")
    print(f"Tokens per second: {tokens_per_sec:.2f}")
else:
    print("Token usage_metadata information is not available for this response.")
