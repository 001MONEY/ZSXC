### wrap_model_call  缠绕式中间件

from langchain.agents.middleware import (wrap_model_call,
                                         ModelRequest,
                                         ModelResponse)

from langchain_ollama import ChatOllama
from typing import Callable
from langchain.agents import create_agent
from langchain.messages import AIMessage

model1 = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434",reasoning= False)


model2 = ChatOllama(model="qwen3:14b",base_url="http://192.168.1.105:11434",reasoning= True)




# #模型的重试+降级
# @wrap_model_call
# def retry_model(
#     request: ModelRequest,
#     handler: Callable[[ModelRequest], ModelResponse],  # handler 是一个函数
# ) -> ModelResponse:
#     for i in range(3):
#         try:
#             res = handler(request,timeout1=10)
#             return res
#         except Exception as e:
#             print(f"模型调用失败，重试第{i+1}次")
#             if i == 2:
#                 print(f"模型调用失败，进行模型降级")

#                 request = request.override(model=model2)
#                 res = handler(request)
#                 return res


# #修改模型的输入与输出
# @wrap_model_call
# def uppercase_responses(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
#     res = handler(request)

#     print(type(res))
#     print(res)

#     res = ModelResponse(result = [AIMessage(content="你不好",response_metadata=res.result[0].response_metadata)],structured_response=res.structured_response,
#     )
#     return res

#修改模型的输出
@wrap_model_call
def uppercase_responses(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    return AIMessage(content="你不好")


agent = create_agent(
    model=model1,
    tools=[],
    middleware=[uppercase_responses]
)


result = agent.invoke({"messages":[{"role": "user", "content": "你好"}]})
print(result)


            

