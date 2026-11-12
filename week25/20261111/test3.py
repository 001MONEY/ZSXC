#Milvus 的使用

from langchain_milvus import Milvus

from langchain_ollama import OllamaEmbeddings
embedding_model = OllamaEmbeddings(model="bge-large:latest",base_url="http://192.168.1.105:11434")

# db = Milvus(
#     embedding_function=embedding_model,
#     connection_args={"uri": "http://192.168.2.197:19530"},
#     collection_name="car_info_collection"
# )


# res = db.similarity_search("领克的保修期是多长",k=3)

# print(res)



texts = [
    "领克的保修期是多长",
    "宝马x5的配置怎么样",
    "奔驰c级的配置怎么样",
    "奥迪a6的配置怎么样",
    "宝马x5的配置怎么样",
    "奔驰c级的配置怎么样",
    "奥迪a6的配置怎么样",
]

vectorstore = Milvus.from_texts(
    texts=texts,
    embedding=embedding_model,
    connection_args={"uri": "http://192.168.2.197:19530"},
    collection_name="car_info"
)