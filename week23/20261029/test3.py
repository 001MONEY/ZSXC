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

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是物理{system_name}老师。"),
    MessagesPlaceholder(variable_name="history"), #消息占位符
    ("human", "{question}")]
)  #需要去区分角色


# messages = chat_prompt.format_messages(question="他有什么特征?",
#                                       history=[("human", "解释一下什么是量子计算?"),("assistant", "量子计算是一种计算方式，它利用量子力学原理进行计算。")],
#                                       system_name="物理")
# print(messages)
# print(type(messages))
# print("--------------------------------------------------------------------------------------")
