#state的的追加与消息列表的使用

from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from operator import add
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage



model = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434")


class State(TypedDict):
    messages:Annotated[list,add]

_builder = StateGraph(State)


def chat(state:State) -> State:

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

res = compiled_graph.invoke({"messages":[HumanMessage(content="你好")]})
print(res)




