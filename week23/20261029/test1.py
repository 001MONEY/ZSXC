#提示词模板

from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="qwen3.8-flash",
    base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
    api_key="sk-ws-H.PXXRYMP.Y6wZ.MEQCIDKz3lBk18q7GfsxvrJoLvFp6TgNTTCPdcuXKsqZisUKAiA1nv-nq4i8djg8xXanm2GsBQ8SJ70dBUIZ4XCsqxl4LA",
    max_tokens=1000,
    temperature=1.2,
    top_p=0.9,
    model_provider="openai"
)


prompt = PromptTemplate(
    template="你好，我的名字是{name}，我今年{age}岁，我擅长{skill}!",
    input_variables=["name","age","skill"],
)

# print(prompt.format(name="World",age=18,skill="编程"))

# print(type(prompt.format(name="World",age=18,skill="编程")))



# print(prompt.invoke({"name":"World","age":18,"skill":"编程"}))

# print(type(prompt.invoke({"name":"World","age":18,"skill":"编程"})))

print(prompt.invoke({"name":"World","age":18,"skill":"编程"}).to_messages())

# print(model.invoke(prompt.invoke({"name":"World","age":18,"skill":"编程"})))



