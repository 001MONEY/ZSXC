from langchain_core.prompts import (PromptTemplate,ChatPromptTemplate,MessagesPlaceholder,
                                    FewShotPromptTemplate,FewShotChatMessagePromptTemplate
                                    )
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
#自己写一个prompt模板

# chat_prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是物理老师。这是一个回答的样例：{example}"),
#     MessagesPlaceholder(variable_name="history"), #消息占位符
#     ("human", "{question}")]
# )  #需要去区分角色


# example = "{'title':'量子计算','content':'量子计算是一种计算方式，它利用量子力学原理进行计算}"


# messages = chat_prompt.format_messages(example=example,question="什么是量子计算?",history=[])


# response = model.invoke(messages)
# print(response.content)


examples = [
    {
        "question": "你好吗?",
        "answer": "师哥，我很好"
    },
    {
        "question": "今天周几?",
        "answer": "师哥，今天星期二"
    },
    {
        "question": "天气好吗?",
        "answer": "师哥，是的，今天天气很好"
    }
]

# example_prompt = PromptTemplate(
#     input_variables=["question", "answer"],
#     template="问题: {question}\n回答: {answer}\n"
# )


# prompt = FewShotPromptTemplate(
#     examples=examples,
#     example_prompt=example_prompt,
#     prefix="你是一个礼貌的助手，总是以'师哥'开头回答问题。以下是示例：\n",
#     suffix="问题: {input}\n回答:",
#     input_variables=["input"]
# )

# messages = prompt.invoke({"input":"你好吗?"})
# print(messages.to_messages())

# print(type(messages.to_messages()))


#使用FewShotChatMessagePromptTemplate

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
    ("ai", "{answer}"),
])


prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)


final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个中二的服务员，请用类似的风格回答问题。"),
        prompt,
        ("human", "{input}"),
    ]
)


print(final_prompt.invoke({"input":"你好吗?"}))