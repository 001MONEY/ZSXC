#回溯机制


from langgraph.graph import StateGraph,START,END

from typing import TypedDict,Annotated

from langgraph.graph.message import add_messages,AnyMessage

from langgraph.checkpoint.memory import InMemorySaver

from langgraph.prebuilt.tool_node import ToolNode  #工具节点 工具节点专门执行工具


from langchain_openai import ChatOpenAI

from langchain.tools import tool

from pydantic import BaseModel,Field

from langchain.messages import ToolMessage


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

#执行图
result = graph.stream({"messages":[("user","What is the weather in Tokyo?")]},config={"configurable":{"thread_id":"user_1"}},
                    stream_mode="values")


for chunk in result:
    print(chunk)
    print("-"*100)

#获取历史
history = list(graph.get_state_history(config={"configurable":{"thread_id":"user_1"}}))



print("历史快照:")


#重新加载了内容后继续执行

# for s in history:
#     print("next=",s.next)
#     print("messages=",s.values["messages"])

#     print("-"*100)

#     if s.next == ("tool_node",):
#         print("选中回放点:",s.next,s.values["messages"])

#         for chunk in graph.stream(None,s.config,stream_mode="values"):
#             print(chunk)
#             print("-"*100)

#         break


#重新加载并且修改内容
for s in history:
    if s.next == ("tool_node",):
        print("选中回放点:",s.next)


        tool_call_id = s.values["messages"][-1].tool_calls[0]["id"]



        fork_config = graph.update_state(s.config,{"messages":[ToolMessage(content="用户拒绝使用工具，直接回答", tool_call_id=tool_call_id)]},as_node="tool_node")


        

        for chunk in graph.stream(None,fork_config,stream_mode="values"):
            print(chunk)
            print("-"*100)
        break












