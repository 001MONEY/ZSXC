from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from pydantic import BaseModel, Field

model = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434")
model_openai = ChatOpenAI(model="qwen3.5:9b",api_key="sk-proj-1234567890",base_url="http://192.168.1.105:11434/v1")




class Weather(BaseModel):
    city: str = Field(description="城市名",example="北京")
    weather: str = Field(description="天气",example="晴")
    temperature: float = Field(description="温度",example=25.0)


@tool(args_schema=Weather)
def get_weather(city: str,weather: str,temperature: float) -> Weather:
    """查询指定城市的当前天气。"""

    return f"查询{city}的天气为{weather}，温度为{temperature}"


# model_with_tools = model.bind_tools([get_weather])  #绑定工具

# res = model_with_tools.invoke("今天北京天气怎么样")
# print(res)




###model.bind_tools([get_weather]) 等价于以下代码

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                    "weather": {"type": "string", "description": "天气"},
                    "temperature": {"type": "number", "description": "温度"}
                },
                "required": ["city"],
            },
        },
    }]


res = model_openai.invoke("今天北京天气怎么样",tools=TOOLS)
print(res)


