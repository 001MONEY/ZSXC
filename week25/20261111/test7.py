# 多问题改写
from langchain_milvus import BM25BuiltInFunction, Milvus

from langchain_ollama import OllamaEmbeddings,ChatOllama

from langchain_classic.retrievers import MultiQueryRetriever

from langchain_core.prompts import PromptTemplate


embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b",base_url="http://192.168.1.105:11434")

model = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434",reasoning=False)


db = Milvus(
    embedding_function=embedding_model,
    connection_args={"uri":"http://192.168.2.197:19530"},
    builtin_function=BM25BuiltInFunction(analyzer_params={"type": "chinese"}),

    vector_field = ["dense", "sparse"],

    collection_name="car_47",)


retriever = db.as_retriever(search_kwargs={
    "k": 5,
    "fetch_k": 20,
    "ranker_type": "rrf",
    "ranker_params": {"k": 60}  # 设置k=60，表示只考虑前60个召回结果
})



CUSTOM_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""你是一个专业的问题改写助手。请为下面的问题生成4个不同的改写版本，
        每个版本应该：
        - 从不同角度表达相同的意图
        - 使用不同的关键词和表达方式
        - 保持问题的核心含义

        每个问题单独一行，不要编号。

        原始问题: {question}

        改写后的问题:""")



multi_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=model,
    prompt=CUSTOM_PROMPT,
)

res = multi_retriever.invoke("领克的保修期")
print(res)