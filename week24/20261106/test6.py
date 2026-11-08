#流式输出 

from langgraph.graph import StateGraph,START,END

from typing import TypedDict,Annotated

from langgraph.graph.message import add_messages,AnyMessage

from langgraph.checkpoint.memory import InMemorySaver

from langgraph.prebuilt.tool_node import ToolNode  #工具节点 工具节点专门执行工具


from langchain_openai import ChatOpenAI

from langchain.tools import tool

from pydantic import BaseModel,Field


model = ChatOpenAI(model="qwen3.8-max",base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
                          api_key="sk-ws-H.PXXRYMP.amZc.MEQCIF-gXiFkLMB2xnc2QFs1_Df0cOe2eft-8gM_WJkTvP6tAiBOayS79OEDUe7fRF5Gu-iSWJ70oIMXB2RVZSKMVIDYDQ"      )


# 1.定义状态
class State(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]


#定义工具

class WeatherResponse(BaseModel):
    city:str = Field(description="The city to get the weather for")


@tool
def get_weather(city:str) -> WeatherResponse:
    """Get the weather for a given city"""
    return "The weather in " + city + " is sunny"

tools = [get_weather]

model_with_tools = model.bind_tools(tools)


#工具加入工具节点ToolNode
tool_node = ToolNode(tools) 


#构建chat节点
def chat_model(state:State):
    response = model_with_tools.invoke(state["messages"])

    return {"messages":[response]}


#构建条件边‘


def condition_edge(state:State):

    if state["messages"][-1].tool_calls:
        return "tool_node"
    else:
        return END

#构建图
builder = StateGraph(State)

builder.add_node("chat",chat_model)
builder.add_node("tool_node",tool_node)
builder.add_edge(START,"chat")
builder.add_conditional_edges("chat",condition_edge)
builder.add_edge("tool_node","chat")
builder.add_edge("chat",END)

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

#执行图 批量输出
# result = graph.invoke({"messages":[("user","What is the weather in Tokyo?")]},config={"configurable":{"thread_id":"user_1"}})
# print(result)


#流式输出 stream_mode="messages"

# result = graph.stream({"messages":[("user","What is the weather in Tokyo?")]},config={"configurable":{"thread_id":"user_1"}},
#                     stream_mode="messages")

# for chunk in result:

#     message,metadata = chunk

#     if message.content:
#         print(message.content,end="",flush=True)


#流式输出 stream_mode="values"


# result = graph.stream({"messages":[("user","What is the weather in Tokyo?")]},config={"configurable":{"thread_id":"user_1"}},
#                     stream_mode="values")


# for chunk in result:

#     print(chunk)

#     print("-"*100)



#流式输出 stream_mode="updates"
result = graph.stream({"messages":[("user","What is the weather in Tokyo?")]},config={"configurable":{"thread_id":"user_1"}},
                    stream_mode="updates")


for chunk in result:
    for node,value in chunk.items():
        print(f"{node}: {value}")
    print("-"*100)













