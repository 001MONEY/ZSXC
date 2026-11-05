##分发

from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from operator import add
from langgraph.types import Send   #Send 是分发的一种方式



class OverallState(TypedDict):
    subject:list[str]
    jokes:Annotated[list[str],add]



_builder = StateGraph(OverallState)


#构建节点

def generate_joke(state:OverallState) -> OverallState:

    return {"jokes": [f"Joke about {state['subject']}"]}


#怎么完成主题的分发

def continue_to_jokes(state:OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state['subject']]



#添加节点

_builder.add_node("generate_joke",generate_joke)

#画边

_builder.add_conditional_edges(START,continue_to_jokes)
_builder.add_edge("generate_joke",END)

compiled_graph = _builder.compile()

res = compiled_graph.invoke({"subject":["cats","dogs"]})
print(res)

