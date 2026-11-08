#Runtime  和Context


from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import MessagesState
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
import uuid

#获取config

from langgraph.config import get_config


#定义Context
from dataclasses import dataclass
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_id:str



_builder = StateGraph(MessagesState)

def call_model(state:MessagesState,runtime:Runtime[Context]):

    user_id = runtime.context.user_id
    print(user_id)

    store = runtime.store  #获取store

    config = get_config()
    print(config)


    namespace = ("memories",user_id)
    store.put(namespace,str(uuid.uuid4()),{"value":"User name is Bob"})
    store.put(namespace,str(uuid.uuid4()),{"value":"User likes Python"})

    memories = store.search(namespace,query="Bob  likes python")
    for d in memories:
        print(d.value["value"])

    return {"messages":["Hello, world!"]}



_builder.add_node("call_model",call_model)

_builder.add_edge(START,"call_model")
_builder.add_edge("call_model",END)

store = InMemoryStore()
checkpointer = InMemorySaver()

graph = _builder.compile(store=store,checkpointer=checkpointer)  #store的使用没有checkpointer也可以

res = graph.invoke({"messages":["Hello, world!"]},config={"configurable":{"thread_id":"user_10086"}},context=Context(user_id="user_1"))
print(res)








