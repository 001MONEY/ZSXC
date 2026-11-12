#混合检索

from langchain_milvus import BM25BuiltInFunction, Milvus

from langchain_ollama import OllamaEmbeddings


from langchain_community.document_loaders  import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter


embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b",base_url="http://192.168.1.105:11434")

loader = PyPDFLoader("/teacher_data/wtb/教学文档/大模型教学文档/large_model/bigmodel/RAG/car_info.pdf")

documents = loader.load()


#切片处理
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=70,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
chunks = text_splitter.split_documents(documents)




#怎么入库以及混合检索

db = Milvus.from_documents(
    documents=chunks[:100],
    embedding=embedding_model,
    connection_args={"uri":"http://192.168.2.197:19530"},

    builtin_function=BM25BuiltInFunction(analyzer_params={"type": "chinese"}),

    vector_field = ["dense", "sparse"],

    collection_name="car_47",

    consistency_level="Session",    # ← 一致性级别放这里

    drop_old=False,  # 追加写入（默认），不破坏已有数据
)


'''
Session	同一会话里更容易立刻读到刚写的数据（常用于实验/写完就查）。
Strong	更强的“立刻对所有人可见”（更严格但通常更慢/更耗）。
Bounded	介于两者之间：有一定延迟上界/折中一致性。
'''