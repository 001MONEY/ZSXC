from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder #消息占位符
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


#适合多轮对话的模板
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是物理{system_name}老师。"),
    ("human", "解释一下什么是量子计算,你好{name}?")]
)  #需要去区分角色


print(chat_prompt)
print("--------------------------------------------------------------------------------------")
print(type(chat_prompt))
print("--------------------------------------------------------------------------------------")


messages = chat_prompt.format(system_name="高级",name="张三")
print(messages)
print(type(messages))

print("--------------------------------------------------------------------------------------")


messages2 = chat_prompt.format_prompt(system_name="高级",name="张三")
print(messages2)
print(type(messages2))

print("--------------------------------------------------------------------------------------")


messages3 = chat_prompt.format_messages(system_name="高级",name="张三")
print(messages3)
print(type(messages3))
print("--------------------------------------------------------------------------------------")

messages4 = chat_prompt.invoke({"system_name":"高级","name":"张三"})
print(messages4)
print(type(messages4))
print("--------------------------------------------------------------------------------------")
