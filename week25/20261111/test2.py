#把检索器转换为工具

from langchain_core.tools import create_retriever_tool

from langchain_community.vectorstores import Chroma

from langchain_ollama import OllamaEmbeddings

embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b",base_url="http://192.168.1.105:11434")

db = Chroma(
    collection_name="car_info",
    embedding_function=embedding_model,
    persist_directory="/teacher_data/wtb/教学文档/大模型教学文档/langchain_47/20261109RAG/chroma_db"
)


res = db.as_retriever(search_kwargs={"k": 3})

result = res.invoke("宝马x5的配置怎么样？")


retriever_tool = create_retriever_tool(res, name="car_info_retriever", description="查询领克汽车的信息")
print(retriever_tool.description)




