from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import ToolMessage,HumanMessage
from pydantic import BaseModel, Field
from langchain_core.tools import tool


model = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434")

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = []
    return store


prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手。"),
    MessagesPlaceholder(variable_name="chat_history")
])

class Weather(BaseModel):
    city: str = Field(description="城市名")
    # weather: str = Field(description="天气")
    # temperature: str = Field(description="温度")


@tool(args_schema=Weather)
def get_weather(city: str,) -> Weather:
    """查询指定城市的当前天气。"""

    return f"查询{city}的天气为晴，温度为25.0"


tools = {"get_weather": get_weather}


tools_list = [get_weather]

model_with_tools = model.bind_tools(tools_list)


chain = prompt | model_with_tools


def chat(session_id: str):
    store_dict = get_session_history(session_id)
    while True:
        
        if len(store_dict[session_id]) > 0 and isinstance(store_dict[session_id][-1],ToolMessage):
                # input_text = "请查看上一论的工具结果"
                # store_dict[session_id].append(HumanMessage(content=input_text))
                pass
        else:
            input_text = input("请输入问题：")
            store_dict[session_id].append(HumanMessage(content=input_text))
        if input == "exit":
            break
        response = chain.invoke({"chat_history": store_dict[session_id]},config={"configurable": {"session_id": session_id}})
        if len(response.tool_calls) > 0:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_result = tools[tool_name].invoke(input=tool_args)
                store_dict[session_id].append(ToolMessage(content=tool_result,tool_call_id=tool_call["id"])) #id同一个工具调用的id相同
        else:
            store_dict[session_id].append(response)
            print(response.content)



if __name__ == "__main__":
    chat("user_001")