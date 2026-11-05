#子图的构建

from langgraph.graph import StateGraph,START,END

from typing import TypedDict,Annotated

from langgraph.graph.message import add_messages,AnyMessage

from langchain.messages import HumanMessage
from operator import add


#1 构建状态
class MessagesState(TypedDict):
    messages:Annotated[list[AnyMessage],add]
    foo:int



class SubState(TypedDict):
    messages:Annotated[list[AnyMessage],add]
    bar:str

#2构建图
_builder = StateGraph(MessagesState)

_sub_builder = StateGraph(SubState)

#3子图的构建

def sub_node1(state:SubState) -> SubState:
    return {"messages":[HumanMessage(content="子图1")]}


def sub_node2(state:SubState) -> SubState:
    return {"messages":[HumanMessage(content="子图2")]}


_sub_builder.add_node("sub_node1",sub_node1)
_sub_builder.add_node("sub_node2",sub_node2)

_sub_builder.add_edge(START,"sub_node1")
_sub_builder.add_edge("sub_node1","sub_node2")
_sub_builder.add_edge("sub_node2",END)

compiled_sub_graph = _sub_builder.compile()


#4主图的构建
def main_node1(state:MessagesState) -> MessagesState:
    return {"messages":[HumanMessage(content="主图1")]}

def main_node2(state:MessagesState) -> MessagesState:
    return {"messages":[HumanMessage(content="主图2")]}


_builder.add_node("main_node1",main_node1)
_builder.add_node("main_node2",main_node2)
_builder.add_node("sub_graph",compiled_sub_graph)

_builder.add_edge(START,"main_node1")
_builder.add_edge("main_node1","sub_graph")
_builder.add_edge("main_node1","main_node2")
_builder.add_edge("main_node2",END)
_builder.add_edge("sub_graph",END)

compiled_graph = _builder.compile()

res = compiled_graph.invoke({"messages":[HumanMessage(content="我开始了")]})

print(res)

from IPython.display import display,Image
compiled_graph.get_graph().draw_mermaid_png(output_file_path="./7.main_graph.png")
compiled_graph.get_graph(xray=1).draw_mermaid_png(output_file_path="./5.main_sub_graph.png")
