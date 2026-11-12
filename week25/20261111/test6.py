#混合检索怎么使用 还有就是 排序的算法

from langchain_milvus import BM25BuiltInFunction, Milvus

from langchain_ollama import OllamaEmbeddings


embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b",base_url="http://192.168.1.105:11434")


db = Milvus(
    embedding_function=embedding_model,
    connection_args={"uri":"http://192.168.2.197:19530"},
    builtin_function=BM25BuiltInFunction(analyzer_params={"type": "chinese"}),

    vector_field = ["dense", "sparse"],

    collection_name="car_47",)



#基于权重的重排
# retriever = db.as_retriever(search_kwargs={
#     "k": 5,
#     "fetch_k": 20,
#     "rank_type": "weighted",
#     "ranker_params": {"weights": [0.6, 0.4]}  # dense=0.6, sparse=0.4
#     })

# '''
# weighted 排序最大的缺陷

# 当一个文档只被 BM25 找到、向量检索没找到时：

# 最终分数 = 0.6 × 0（向量分数缺失，补0） + 0.4 × BM25归一化分数
#          = 0.4 × BM25归一化分数

# 这个文档直接损失了 60% 的权重，
# 严重被压低，很可能排不进最终的 top-k，即使它在语义上非常相关。

# '''

# res = retriever.invoke("车顶行李架")
# print(res)


#基于rrf的重排
# RRF：不看原始分数，只看排名位置，用"倒数+平滑"把多路召回的结果公平融合，排名越靠前贡献越大，无需调参即插即用。


retriever = db.as_retriever(search_kwargs={
    "k": 10,
    "fetch_k": 20,
    "rank_type": "rrf",
    "ranker_params": {"k":60}
    })


# '''

# 稠密检索给出自己的排名列表：文档A第1，文档B第2，文档C第3...
# BM25 给出自己的排名列表：文档C第1，文档A第3，文档D第2...
# RRF 用 1/(60 + rank) 把每个文档在每一路的贡献分加起来，得到融合分数
# 按融合分数重新排列，得到最终结果

#  取倒数 计算贡献度
# | 排名 | 计算         | 贡献     |
# | -- | ---------- | ------ |
# | 1  | 1/1 = 1.0  | 太大 |
# | 2  | 1/2 = 0.5  | 直接减半   |
# | 10 | 1/10 = 0.1 | 只剩10%  |

# 问题：排名1和2差距太大（1.0 vs 0.5），不公平

# 取平滑指数 k=60

# | 排名 | 计算                | 贡献        |
# | -- | ----------------- | --------- |
# | 1  | 1/(60+1) = 1/61   | **0.016** |
# | 2  | 1/(60+2) = 1/62   | **0.016** |
# | 10 | 1/(60+10) = 1/70  | **0.014** |
# | 60 | 1/(60+60) = 1/120 | **0.008** |


# 效果：差距变小了，Top-60都有价值，60名后才明显衰减
# '''


# print(retriever.invoke("车顶行李架"))


#rerank

# 使用 LangChain 的包装类构建重排序器
# pip install sentence-transformers -i https://mirrors.aliyun.com/pypi/simple/


from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from langchain_classic.retrievers import ContextualCompressionRetriever


model = HuggingFaceCrossEncoder(model_name="/teacher_train/wtb/hug_model/rerank_qwen/")

#构建重排序器

compressor = CrossEncoderReranker(model=model, top_n=3)

#构建压缩检索器

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)


print(compression_retriever.invoke("车顶行李架"))



