from typing import Annotated, TypedDict
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import create_chat_model
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    messages: Annotated[list, add_messages]


builder = StateGraph(State)

model = create_chat_model()


def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


# Add the chatbot node
builder.add_node("chatbot", chatbot)

# Add edges
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# Run the graph
input = {"messages": [HumanMessage("hi!")]}
for chunk in graph.stream(input):
    print(chunk)
