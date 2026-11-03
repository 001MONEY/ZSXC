from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.messages import AIMessage,HumanMessage
from pydantic import BaseModel, Field

class Weather(BaseModel):
    city: str = Field(description="城市名")
    weather: str = Field(description="天气")
    temperature: float = Field(description="温度")

class Add(BaseModel):
    a: float = Field(description="第一个数")
    b: float = Field(description="第二个数")

model = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434")


#大模型的工具本质上就是函数。所以我们可以直接定义函数，然后作为工具传给大模型。
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。"""
    return f"{city}：晴，25°C"

def add(a: float, b: float) -> str:
    """计算两个数的和。"""
    return str(a + b)


tools = {"get_weather": get_weather,"add": add}



class tool_call(BaseModel):
    tool_name: str = Field(description="工具名")
    tool_args: dict = Field(description="工具参数")
    output: str | None = Field(description="不需要工具时写给人的回复；需要调用工具时填 null")


output_parser = PydanticOutputParser(pydantic_object=tool_call)

message_history = []


prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个聊天助手{tool_call_instructions}。你有以下工具：get_weather和add。当需要查询天气的时候使用get_weather工具，它需要的参数格式为：{{'city': '城市名'}}，当需要计算两个数的和的时候使用add工具，它需要的参数格式为：{{'a': 1, 'b': 2}}。"),
    MessagesPlaceholder(variable_name="message_history"),
    ("human", "{inputs_text}"),
])




chain = prompt | model | output_parser


#多轮对话带工具使用
def chat(inputs_text: str,message_history: list):
    response = chain.invoke({"inputs_text": inputs_text,"tool_call_instructions": output_parser.get_format_instructions(),"message_history": message_history})

    if response.output is None or response.output == "None":
        tool_call_result = tools[response.tool_name](**response.tool_args)
        message_history.append(AIMessage(content=f"这是需要调用的工具的名称：{response.tool_name}，它的参数是：{response.tool_args}"))
        message_history.append(HumanMessage(content=f"这是工具的返回结果：{tool_call_result}"))
        return None

    message_history.append(AIMessage(content=response.output))
    return response.output




def main():
    while True:
        inputs_text = input("请输入：")
        response = chat(inputs_text,message_history)
        print(response)
        if response is None:
            print("需要调用工具")
            response = chat("这是工具的结果",message_history)
            print(response)

if __name__ == "__main__":
    main()