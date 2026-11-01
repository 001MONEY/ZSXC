from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

model = init_chat_model(
    model="qwen3.8-max",
    base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
    api_key="sk-ws-H.PXXRYMP.Y6wZ.MEQCIDKz3lBk18q7GfsxvrJoLvFp6TgNTTCPdcuXKsqZisUKAiA1nv-nq4i8djg8xXanm2GsBQ8SJ70dBUIZ4XCsqxl4LA",
    max_tokens=1000,
    temperature=1.2,
    top_p=0.9,
    model_provider="openai",
    extra_body={"enable_thinking": False})



#1.构建一个空字典

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 2. prompt 里用 MessagesPlaceholder 放历史消息
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的AI助手"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])


chain = prompt | model



#3.包装成带记忆的链  缠绕式
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


res1 = chain_with_history.invoke({"input": "你好，我的名字是张三，我是你的老板"},config={"configurable": {"session_id": "user-001"}})

print(res1)



res2 = chain_with_history.invoke({"input": "我叫什么？"},config={"configurable": {"session_id": "user-001"}})

print(res2)
