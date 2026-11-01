from langchain_core.prompts import (PromptTemplate)
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="qwen3.8-flash",
    base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
    api_key="sk-ws-H.PXXRYMP.Y6wZ.MEQCIDKz3lBk18q7GfsxvrJoLvFp6TgNTTCPdcuXKsqZisUKAiA1nv-nq4i8djg8xXanm2GsBQ8SJ70dBUIZ4XCsqxl4LA",
    max_tokens=1000,
    temperature=1.2,
    top_p=0.9,
    model_provider="openai")



prompt = PromptTemplate(
    input_variables=["question"],
    template="你是一个物理老师，请回答问题：{question}"
)


chain = prompt | model

print(chain.invoke({"question": "什么是相对论？"}))