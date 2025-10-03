from typing import TypedDict, Annotated, Sequence
import asyncio
from contextlib import aclosing

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

    event = asyncio.Event()

    user_input = {
        "messages": [
            HumanMessage(
                "How old was the 30th president of the United States when he died?"
            )
        ]
    }

    config = {"configurable": {"thread_id": "1"}}

    async with aclosing(graph.astream(user_input, config)) as stream:
        async for chunk in stream:
            if event.is_set():
                break
            else:
                print(chunk)

    # Simulate interruption after 2 seconds (note: for a real interruption,
    # set the event from another task while the loop is running)
    await asyncio.sleep(2)
    event.set()


if __name__ == "__main__":
    asyncio.run(main())
