#大模型的 结构化输出 ，json 格式输出


#1.使用langchian的输出解释器，然后手动的加上提示词，让大模型输出json格式
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import PromptTemplate

# llm = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434")

# parser = JsonOutputParser()
# parser_str = parser.get_format_instructions()
# prompt = PromptTemplate(
#     input_variables=["question","parser_str"],
#     template="What is the capital of {question}?{parser_str}")

# chain = prompt | llm | parser

# result = chain.invoke({"question":"China","parser_str":parser_str})
# print(result)



#2.依赖于模型的function calling 功能，直接让大模型输出json格式，不需要手动的加上提示词
# from langchain_ollama import ChatOllama
# from pydantic import BaseModel,Field

# class output_parser(BaseModel):
#     weather: str = Field(description="天气预报")
#     window : str = Field(description="风力大小")
#     temperature: int = Field(description="温度")

# llm = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434",reasoning= False)


# llm_with_output_parser = llm.with_structured_output(output_parser,method="function_calling")
# result = llm_with_output_parser.invoke("旧金山的天气很好，小雨，20度，东南风")
# print(result)

# print(type(result))

# print(result.model_dump_json())



#第三种厂商层面做的约束，保证模型 token 采样时只能生成符合 json schema的内容 百分百生成json格式但需要推理厂商支持



# from langchain_openai import ChatOpenAI
# from pydantic import BaseModel,Field

# llm = ChatOpenAI(
#     model="qwen3-max",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     api_key="sk-ws-H.RYLEXRL.gLta.MEYCIQCcevt5KrUbUtc_N5JKOTHW81fzvBlrJYF4XBw_mzN6AgIhAMAHz6-dE8PU2FRqm4-uOqUimgEdVcaWo-q3Z5HC97Xg",
#     extra_body={"enable_thinking": False}
# )

# class output_parser(BaseModel):
#     weather: str = Field(description="天气预报")
#     window : str = Field(description="风力大小")
#     temperature: int = Field(description="温度")



# llm = llm.with_structured_output(output_parser,method="json_schema")

# res = llm.invoke("旧金山的天气很好，小雨，20度，东南风")

# print(res)

#在create_agent方法中，直接指定json输出格式


from langchain_ollama import ChatOllama

from pydantic import BaseModel,Field

from langchain.agents import create_agent
from langchain.tools import tool

from langchain.agents.structured_output import ToolStrategy,ProviderStrategy


model = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434",reasoning= True)

class output_parser(BaseModel):
    weather: str = Field(description="天气预报")
    window : str = Field(description="风力大小")
    temperature: int = Field(description="温度")



@tool
def check_weather(location: str):
    '''返回指定位置的天气预报'''
    return f"{location} 总是阳光明媚,有风,温度20度"



graph = create_agent(
    model=model,
    tools=[check_weather],
    # response_format=ToolStrategy(output_parser),#包装成工具
    response_format=ProviderStrategy(output_parser),#包装成提供者
)


result = graph.invoke({"messages":[{"role": "user", "content": "旧金山的天气怎么样"}]})
print(result.get("structured_response"))





