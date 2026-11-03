from langchain.agents import create_agent

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from langgraph.checkpoint.memory import MemorySaver

model = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434")

class Weather(BaseModel):
    city: str = Field(description="城市名")
    # weather: str = Field(description="天气")
    # temperature: str = Field(description="温度")



@tool(args_schema=Weather)
def get_weather(city: str,) -> Weather:
    """查询指定城市的当前天气。"""

    return f"查询{city}的天气为晴，温度为25.0"

graph = create_agent(
            model=model,
            tools=[get_weather],
            system_prompt="You are a helpful assistant",
            checkpointer=MemorySaver(),
        )



def chat():
    while True:
        user_input = input("请输入: ")
        if user_input == "exit":
            break
        response = graph.invoke({"messages": [{"role": "user", "content": user_input}]},config={"configurable": {"thread_id": "user_001"}})
        print(response)


if __name__ == "__main__":
    chat()
