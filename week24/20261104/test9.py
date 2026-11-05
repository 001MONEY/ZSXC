#检查点的使用

from langgraph.graph import StateGraph,START,END

from typing import TypedDict,Annotated

from langgraph.graph.message import add_messages  , AnyMessage
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.redis import RedisSaver

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

#内存检查点

# checkpointer = InMemorySaver()

# compiled_graph = _builder.compile(checkpointer=checkpointer)

# res = compiled_graph.invoke({"messages":["你好,我是黑大帅"]},config={"configurable":{"thread_id":"123"}})
# # print(res)


# res2 = compiled_graph.invoke({"messages":["你知道我是谁吗"]},config={"configurable":{"thread_id":"123"}})
# # print(res2)



# history = compiled_graph.get_state_history(config={"configurable":{"thread_id":"123"}})


# for i,state in enumerate(history):

#     print(f"历史快照{i}: {state}")

#     print("-"*100)

# print("历史快照结束")


#redis检查点

DB_URI = "redis://192.168.2.197:6379/0"

with RedisSaver.from_conn_string(DB_URI) as checkpointer:  #帮你关闭链接

    checkpointer.setup() #它才会在 Redis 里创建 checkpoint 和 checkpoint_write 两个索引

    compiled_graph = _builder.compile(checkpointer=checkpointer)

    res = compiled_graph.invoke({"messages":["你好,我是黑大帅"]},config={"configurable":{"thread_id":"123"}})
    print(res)

    res2 = compiled_graph.invoke({"messages":["你知道我是谁吗"]},config={"configurable":{"thread_id":"123"}})
    print(res2)

    history = compiled_graph.get_state_history(config={"configurable":{"thread_id":"123"}})

    for i,state in enumerate(history):

        print(f"历史快照{i}: {state.next}")
        print("-"*100)



