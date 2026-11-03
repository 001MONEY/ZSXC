from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

model = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434")

@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。"""
    return f"{city}：晴，25°C"

@tool
def add(a: float, b: float) -> str:
    """计算两个数的和。"""
    return str(a + b)


tools = {"get_weather": get_weather,"add": add}


res = get_weather.invoke("北京")
print(res)

print(get_weather.description)
print(get_weather.args)


model_with_tools = model.bind_tools([get_weather])

res = model_with_tools.invoke("今天北京天气怎么样")
print(res)

if res.tool_calls:
    for tool_call in res.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_result = tools[tool_name].invoke(input=tool_args)
        print(tool_result)
        print(type(tool_result))
