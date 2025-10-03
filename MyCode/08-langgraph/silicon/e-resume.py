from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    # Chat history across nodes
    messages: Annotated[Sequence[BaseMessage], add_messages]


def answer_node(state: State) -> dict:
    # Minimal deterministic node
    question = ""
    for m in state.get("messages", []):
        if isinstance(m, HumanMessage):
            question = m.content
    answer = "I'm not sure."
    if "30th president" in question.lower():
        answer = "Calvin Coolidge died at age 60."
    return {"messages": [AIMessage(content=answer)]}


async def main():
    # Build a simple graph with a valid state schema
    builder = StateGraph(State)
    builder.add_node("answer", answer_node)
    builder.set_entry_point("answer")
    graph = builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "1"}}

    # Try to resume: if there is no prior state for this thread, seed with an initial input
    state = graph.get_state(config)
    has_prior = False
    try:
        # state.values is the saved state dict in LangGraph
        has_prior = bool(getattr(state, "values", None))
    except Exception:
        has_prior = False

    initial_input = None if has_prior else {
        "messages": [
            HumanMessage(
                "How old was the 30th president of the United States when he died?"
            )
        ]
    }

    # Resume from prior state if any; otherwise start with the seeded input
    output = graph.astream(initial_input, config, interrupt_before=["tools"])

    async for c in output:
        print(c)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
