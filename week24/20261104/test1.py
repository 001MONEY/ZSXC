#最简单的图
from langgraph.graph import StateGraph,START,END
from typing import TypedDict


#定义我们的载体(状态)(State)

class State(TypedDict):
    foo:int
    bar:str



#1.构建我们的画纸
graph = StateGraph(State)



#2.构建我们的节点


def node1(state:State) -> State:

    foo = state["foo"]
    print(f"foo: {foo}")

    foo = foo + 1

    return {"foo":foo}


#3.添加节点
graph.add_node("node1",node1)


#4.添加边(无条件边连接)

graph.add_edge(START, "node1")
graph.add_edge("node1", END)


#5.编译我们的图

compiled_graph = graph.compile()

#6.执行我们的图

res = compiled_graph.invoke({"foo":0,"bar":"hello"})

print(res)










    





