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

    user_input = {
        "messages": [
            HumanMessage(
                "How old was the 30th president of the United States when he died?"
            )
        ]
    }

    config = {"configurable": {"thread_id": "1"}}

    output = graph.astream(user_input, config, interrupt_before=["tools"])

    async for c in output:
        print(c)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
