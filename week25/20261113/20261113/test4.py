###wrap_tool_call中间键
from langchain.agents.middleware import (wrap_tool_call)

from langchain.tools.tool_node import ToolCallRequest
from langchain.messages import ToolMessage

from langchain.tools import tool

from langchain_ollama import ChatOllama
from typing import Callable
from langchain.agents import create_agent
import time
from dataclasses import dataclass

model1 = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434",reasoning= False)

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"城市{city}的天气是晴天"


# @wrap_tool_call
# def log_tool_call(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage]) -> ToolMessage:

#     start_time = time.time()
#     res = handler(request)
#     end_time = time.time()
#     print(f"工具调用时间: {end_time - start_time}秒",f"工具调用结果: {res}")
#     return res



BLOCKED_TOOLS = {"user1":{"delete_file", "get_weather"}}  # 禁止调用的工具:
#工具权限校验
@wrap_tool_call

def check_tool_permission(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage]) -> ToolMessage:

    user_id = request.runtime.context.user_id
    print(request)

    if user_id in BLOCKED_TOOLS and request.tool_call["name"] in BLOCKED_TOOLS[user_id]:

        return ToolMessage(content=f"用户{user_id}无权限调用工具{request.tool_call['name']}",
        tool_call_id=request.tool_call["id"])
    
    return handler(request)



@dataclass
class UserContext:
    user_id: str


agent = create_agent(
    model=model1,
    tools=[get_weather],
    middleware=[check_tool_permission],
    context_schema = UserContext

)

result = agent.invoke({"messages":[{"role": "user", "content": "北京天气如何？"}]},
                 context=UserContext(user_id="user1"))
print(result)