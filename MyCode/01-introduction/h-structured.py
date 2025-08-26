from common import (
    create_chat_model,
    parse_json_response,
    create_structured_prompt,
    print_debug_info,
    AnswerWithJustification,
    create_structured_prompt,
)

llm = create_chat_model(model_name="openai/gpt-oss-20b", temperature=0)

# Create structured prompt using common function
question = "What weighs more, a pound of bricks or a pound of feathers?"
response_schema = {
    "answer": "your answer here",
    "justification": "your justification here",
}

prompt = create_structured_prompt(question, response_schema)

# Use invoke instead of with_structured_output since LM Studio doesn't support structured output API
response = llm.invoke(prompt)
print_debug_info(response)

# Parse JSON response using common function
success, result = parse_json_response(response.content, AnswerWithJustification)

if success:
    print("Structured response:", result)
else:
    print(f"Failed to parse response: {result}")
