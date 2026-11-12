#RAGAS测试

from langchain_milvus import BM25BuiltInFunction, Milvus

from langchain_ollama import OllamaEmbeddings,ChatOllama

from openai import AsyncOpenAI
from ragas.llms import llm_factory

from ragas.metrics.collections import ContextRecall, Faithfulness

from langchain_core.prompts import PromptTemplate


#模型的准备
embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b",base_url="http://192.168.1.105:11434")

answer_llm = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434",reasoning=False)

judge = llm_factory(
    "qwen3.5:9b",
    client=AsyncOpenAI(base_url="http://192.168.1.105:11434/v1", api_key="ollama"),
    extra_body={"reasoning_effort": "none"},
    max_tokens=8192,
)


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
    "ranker_params": {"k": 60}
})



faithfulness = Faithfulness(llm=judge)
context_recall = ContextRecall(llm=judge)


#这些是模拟的标注数据
test_questions = [
    {
        "question": "领克汽车车顶行李架的最大载重是多少？",
        "ground_truth": "领克汽车车顶行李架的最大载重是70公斤",
    },
    {
        "question": "领克汽车的保修期是多久？",
        "ground_truth": "领克汽车提供3年或10万公里的整车质保",
    },
    {
        "question": "领克汽车的发动机类型是什么？",
        "ground_truth": "领克汽车搭载2.0T涡轮增压发动机",
    },
]

#构建一个提示词

answer_prompt = PromptTemplate(
    input_variables=["question", "context_text"],
    template= """基于以下上下文信息回答问题。如果上下文中没有相关信息，请说"无法从提供的信息中回答"。

上下文信息：
{context_text}

问题：{question}

答案："""
)

faith_scores = []
recall_scores = []


for item in test_questions:
    question = item["question"]
    ground_truth = item["ground_truth"]

    print(f"问题：{question}")

    retrieved_docs = retriever.invoke(question)

    retrieved_contexts = [doc.page_content for doc in retrieved_docs]


    response = answer_llm.invoke(answer_prompt.format(question=question, context_text="\n\n".join(retrieved_contexts))).content

    print(response)


    faith_score = faithfulness.score(user_input=question, response=response,  retrieved_contexts=retrieved_contexts,).value

    recall_score = context_recall.score(user_input=question, reference=ground_truth, retrieved_contexts=retrieved_contexts,).value


    faith_scores.append(faith_score)
    recall_scores.append(recall_score)
    print({"faithfulness": faith_score, "context_recall": recall_score})



print(faith_scores)
print("--------------------------------")
print(recall_scores)
print("--------------------------------")

















