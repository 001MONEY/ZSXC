#store的简单样例


from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import MessagesState
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
import uuid

_builder = StateGraph(MessagesState)

def call_model(state:MessagesState,store:BaseStore):

    namespace = ("memories","user_1")
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

res = graph.invoke({"messages":["Hello, world!"]},config={"configurable":{"thread_id":"user_1"}})
print(res)






