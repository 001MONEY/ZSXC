#按需获取样例
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate

from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaEmbeddings


import numpy as np


embedding = OllamaEmbeddings(model="bge-large",base_url="http://192.168.1.105:11434")

examples = [
    {"question": "你好吗?", "answer": "师哥，我很好"},
    {"question": "今天周几?", "answer": "师哥，今天星期二"},
    {"question": "天气好吗?", "answer": "师哥，是的，今天天气很好"},
    {"question": "你叫什么?", "answer": "师哥，我是AI助手"},
    {"question": "几点了?", "answer": "师哥，我无法获取当前时间"},
    {"question": "什么是卷积神经网络", "answer": "卷积神经网络就是一个滤波器"},
    {"question": "解释一下机器学习", "answer": "师哥，机器学习是人工智能的一个分支..."},
]



example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="问题: {question}\n回答: {answer}\n"
)


semantic_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=embedding,
    vectorstore_cls=FAISS,
    k=2
)

question = "卷积神经网络"

# print(semantic_selector.select_examples({"question": question}))


semantic_prompt = FewShotPromptTemplate(
            example_selector=semantic_selector,
            example_prompt=example_prompt,
            prefix="基于相似问题，请回答：\n",
            suffix="问题: {question}\n回答:",
            input_variables=["question"]
        )

print(semantic_prompt.invoke({"question": question}))
        