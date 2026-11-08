#在工具中获取config,store,context

from langgraph.graph import StateGraph,START,END

from langgraph.graph.message import MessagesState

from langchain_openai import ChatOpenAI

from langgraph.prebuilt import ToolNode

from langchain.tools import tool,ToolRuntime

from pydantic import BaseModel,Field

from langgraph.runtime import Runtime

from langgraph.checkpoint.memory import InMemorySaver
#store

from langgraph.store.memory import InMemoryStore

store = InMemoryStore()


#context

from dataclasses import dataclass
@dataclass
class Context:
    user_id:str



#定义模型

model = ChatOpenAI(model="qwen3.8-max",base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
                          api_key="sk-ws-H.PXXRYMP.amZc.MEQCIF-gXiFkLMB2xnc2QFs1_Df0cOe2eft-8gM_WJkTvP6tAiBOayS79OEDUe7fRF5Gu-iSWJ70oIMXB2RVZSKMVIDYDQ")


#定义工具

class WeatherResponse(BaseModel):
    city:str = Field(description="The city to get the weather for")



#ToolRuntime是工具的运行时环境，包含了config,store,context ,目的是 权限校验，防止工具被滥用
@tool  
def get_weather(city:str,runtime:ToolRuntime[Context]) -> WeatherResponse:
    """Get the weather for a given city"""
    config = runtime.config
    store = runtime.store
    context = runtime.context
    print(f"config: {config}")
    print(f"store: {store}")
    print(f"context: {context}")
    return "The weather in " + city + " is sunny"



llm_with_tools = model.bind_tools([get_weather])


tool_node = ToolNode(tools=[get_weather])



def call_model(state:MessagesState,runtime:Runtime[Context]) -> MessagesState:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def condition_edge(state:MessagesState) -> str:
    if state["messages"][-1].tool_calls:
        return "tool_node"
    else:
        return END


builder = StateGraph(MessagesState)
builder.add_node("call_model",call_model)
builder.add_node("tool_node",tool_node)
builder.add_edge(START,"call_model")
builder.add_conditional_edges("call_model",condition_edge)
builder.add_edge("tool_node","call_model")
builder.add_edge("call_model",END)

checkpointer = InMemorySaver()

graph = builder.compile(store=store,checkpointer=checkpointer)


res = graph.invoke({"messages":["What is the weather in Beijing?"]},config={"configurable":{"thread_id":"user_1"}},context=Context(user_id="user_1"))






