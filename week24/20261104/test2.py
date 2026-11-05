#有条件边的图
from langgraph.graph import StateGraph,START,END
from typing import TypedDict

#1.定义我们的载体(状态)(State)
class State(TypedDict):
    foo:int


#2.构建画纸

builder = StateGraph(State)


#3.构建节点

def node1(state:State) -> State:

    foo = state["foo"]
    foo = foo + 10
    return {"foo":foo}




def node2(state:State) -> State:
    foo = state["foo"]
    foo = foo -10
    return {"foo":foo}



def node3(state:State) -> State:
    print("node3")
    foo = state["foo"]
    foo = foo*10
    return {"foo":foo}

def node4(state:State) -> State:
    print("node4")
    foo = state["foo"]
    foo = foo/10
    return {"foo":foo}

#定义条件边

def cond_edge(state:State) -> str:

    if state["foo"] > 10:
        return "node3"
    else:
        return "node2"


#4.添加节点
builder.add_node("node1",node1)
builder.add_node("node2",node2)
builder.add_node("node3",node3)
builder.add_node("node4",node4)
#5.添加条件边

builder.add_edge(START,"node1")


builder.add_conditional_edges("node1",cond_edge,["node2","node3"])

builder.add_edge("node2","node4")

builder.add_edge("node3","node4")

builder.add_edge("node4",END)


#6.编译图
compiled_graph = builder.compile()

#7.执行图
res = compiled_graph.invoke({"foo":-10})
print(res)



from IPython.display import Image,display
# display(Image(_graph.get_graph().draw_mermaid_png(output_file_path="1111.png")))

compiled_graph.get_graph().draw_mermaid_png(output_file_path="./con_edge.png")


