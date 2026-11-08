#人工断点 模型可以自己判断进入的节点


from langgraph.graph import StateGraph,START,END

from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph.message import MessagesState

from langchain_openai import ChatOpenAI


from langgraph.types import interrupt, Command

from langgraph.prebuilt.tool_node import ToolNode

from langchain.tools import tool

from pydantic import BaseModel,Field



model = ChatOpenAI(model="qwen3.8-max",base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
                          api_key="sk-ws-H.PXXRYMP.amZc.MEQCIF-gXiFkLMB2xnc2QFs1_Df0cOe2eft-8gM_WJkTvP6tAiBOayS79OEDUe7fRF5Gu-iSWJ70oIMXB2RVZSKMVIDYDQ")



class WeatherResponse(BaseModel):
    city:str = Field(description="The city to get the weather for")


@tool(args_schema=WeatherResponse)
def get_weather(city:str) -> WeatherResponse:
    """Get the weather for a given city"""
    return "The weather in " + city + " is sunny"


class HumanState(BaseModel):
    bools:bool = Field(description="是否需要人工介入")

@tool(args_schema=HumanState)
def human_decision(bools:bool) -> HumanState:
    """对于不知道的问题可以申请人工介入"""

    decision = interrupt("是否需要人工介入？")
    return decision


tool_node = ToolNode(tools=[get_weather,human_decision])


model_with_tools = model.bind_tools(tools=[get_weather,human_decision])


def chat_model(state:MessagesState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def condition_edge(state:MessagesState):
    if state["messages"][-1].tool_calls:
        return "tool_node"
    else:
        return END


builder = StateGraph(MessagesState)
builder.add_node("chat",chat_model)
builder.add_node("tool_node",tool_node)
builder.add_edge(START,"chat")
builder.add_conditional_edges("chat",condition_edge)
builder.add_edge("tool_node","chat")
builder.add_edge("chat",END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


result = graph.stream({"messages":[("user","请问今天成都到西安的车票还有没")]},config={"configurable":{"thread_id":"user_1"}},
                    stream_mode="values")

for chunk in result:
    print(chunk)
    print("-"*100)

snapshot = graph.get_state(config={"configurable":{"thread_id":"user_1"}})



if snapshot.values["messages"][-1].tool_calls and snapshot.values["messages"][-1].tool_calls[0]["name"] == "human_decision":


    for chunk in graph.stream(Command(resume="yes"),config={"configurable":{"thread_id":"user_1"}},stream_mode="values"):
        print(chunk)
        print("-"*100)

        
