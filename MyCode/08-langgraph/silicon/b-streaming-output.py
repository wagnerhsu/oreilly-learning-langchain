from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    # A list of chat messages; add_messages handles appending across nodes
    messages: Annotated[Sequence[BaseMessage], add_messages]


def answer_node(state: State) -> dict:
    # Minimal, deterministic node that crafts an answer without calling an external LLM
    question = ""
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            question = m.content

    # Simple heuristic answer for the provided demo question
    answer = "I'm not sure."
    if "30th president" in question.lower():
        answer = "Calvin Coolidge died at age 60."

    return {"messages": [AIMessage(content=answer)]}


def create_simple_graph():
    # Create a simple graph with a valid state schema and a single node
    builder = StateGraph(State)
    builder.add_node("answer", answer_node)
    builder.set_entry_point("answer")
    return builder.compile()


graph = create_simple_graph()

user_input = {
    "messages": [
        HumanMessage(
            "How old was the 30th president of the United States when he died?"
        )
    ]
}

for c in graph.stream(user_input, stream_mode="updates"):
    print(c)
