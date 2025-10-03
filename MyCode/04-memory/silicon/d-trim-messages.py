from dotenv import load_dotenv
import os

load_dotenv()
base_url = os.environ.get("BASE_URL")
api_key = os.environ.get("API_KEY")
model_name = os.environ.get("MODEL")

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    trim_messages,
)

# Removed ChatOpenAI-based token counting to avoid NotImplementedError when the model isn't known to the tokenizer.
# We'll provide a robust token counter that uses tiktoken if available, and a heuristic fallback otherwise.
try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tiktoken = None


def _count_tokens_messages(messages, model: str | None = None) -> int:
    """Return an estimated token count for a list of LangChain messages.

    - If tiktoken is available, use a best-effort encoding (model-specific when possible, else cl100k_base).
    - Otherwise, fall back to a simple heuristic (~4 chars per token).
    """
    # Build a simple ChatML-like text to approximate tokenization per message
    parts: list[str] = []
    for m in messages:
        role = getattr(m, "type", None) or m.__class__.__name__.replace("Message", "").lower()
        content = m.content
        if isinstance(content, list):
            # Some messages may be structured; coerce to string
            content = " ".join(str(c) for c in content)
        elif content is None:
            content = ""
        parts.append(f"{role}: {content}")
    text = "\n".join(parts)

    # tiktoken-based precise-ish count
    if tiktoken is not None:
        enc = None
        try:
            if model:
                try:
                    enc = tiktoken.encoding_for_model(model)
                except Exception:
                    enc = None
            if enc is None:
                # Default modern chat encoding used by GPT-4/3.5 families
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # If anything fails inside tiktoken, fall back to heuristic
            pass

    # Heuristic: ~4 characters per token (OpenAI guidance)
    approx = max(1, (len(text) // 4))
    return approx


# Define sample messages
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

# Create trimmer (use our custom token counter to avoid NotImplementedError)
trimmer = trim_messages(
    max_tokens=1024,
    strategy="last",
    token_counter=lambda msgs: _count_tokens_messages(msgs, model_name),
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Apply trimming
trimmed = trimmer.invoke(messages)
print(trimmed)
