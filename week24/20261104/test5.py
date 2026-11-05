#MessagesState(图中放的state)

from langgraph.graph import StateGraph,START,END

from typing import TypedDict,Annotated

from langgraph.graph.message import add_messages  , AnyMessage
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage


model = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434")

class MessagesState(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]


_builder = StateGraph(MessagesState)


def chat(state:MessagesState) -> MessagesState:

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个聊天机器人，请根据用户的问题回答"),
        MessagesPlaceholder(variable_name="messages")
    ])

    chain = prompt | model

    res = chain.invoke({"messages":state["messages"]})

    return {"messages":[res]}



_builder.add_node("chat",chat)

_builder.add_edge(START,"chat")
_builder.add_edge("chat",END)

compiled_graph = _builder.compile()

res = compiled_graph.invoke({"messages":["你好"]})
print(res)

