# 将数据保存到向量数据库中

# from langchain_community.vectorstores import Chroma
# from langchain_ollama import OllamaEmbeddings

# embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b",base_url="http://192.168.1.105:11434")




# 将数据保存到向量数据库中  文本数据
# datas=[
#         "小明喜欢吃苹果", 
#         "小红喜欢吃榴莲",
#         "小明的女朋友是小丽",
#         "王老师是一个好老师",
#         "小李喜欢吃香蕉",
#         "小王的男朋友是大帅哥"
#     ]

# db = Chroma.from_texts(
#     texts=datas,
#     embedding=embedding_model,
#     collection_name="text_data",   # 集合名称
#     persist_directory="/teacher_data/wtb/教学文档/大模型教学文档/langchain_47/20261109RAG/chroma_db"  # 数据库存储路径

# )


# query="谁是老师"
# results = db.similarity_search_with_score(query,k=2)
# print(results)


#使用已经保存到向量数据库中的数据进行查询

# db = Chroma(
#     collection_name="text_data",
#     embedding_function=embedding_model,
#     persist_directory="/teacher_data/wtb/教学文档/大模型教学文档/langchain_47/20261109RAG/chroma_db"
# )

# query="谁是老师"
# results = db.similarity_search(query,k=2)
# print(results)



# 将数据保存到向量数据库中  document数据

from langchain_community.document_loaders import PyMuPDFLoader
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

#加载嵌入模型，将文档转换为向量

embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b",base_url="http://192.168.1.105:11434")

#加载pdf文件
loader = PyMuPDFLoader(file_path="/teacher_data/wtb/教学文档/大模型教学文档/large_model/bigmodel/data/car_info.pdf")
docs = loader.load()


#数据清洗·

def clean_pdf_text_re(text: str) -> str:
    _SECTION_HEADERS = "提示信息|警告！|注意！|说明！|帮助信息|图标和插图说明"
    """清洗 PDF 提取文本中的排版折行，保留段落/标题结构。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 合并汉字之间的排版折行（如 造\n成、版\n本）
    text = re.sub(r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])", "", text)

    # 合并标点后的折行（如 部位，\n其他）
    text = re.sub(r"(?<=[，。；：！？、）」』])\n(?=[\u4e00-\u9fff])", "", text)

    # 在常见标题词后恢复换行，避免「提示信息警告！」粘在一起
    text = re.sub(rf"({_SECTION_HEADERS})(?=[\u4e00-\u9fff])", r"\1\n", text)

    # 多个空行压成段落分隔
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



from langchain_core.documents import Document


cleaned_documents = []  #清洗后的文档


for doc in docs:

    cleaned_content = clean_pdf_text_re(doc.page_content)  #纯文

    cleaned_doc = Document(
        page_content=cleaned_content,
        metadata=doc.metadata  #保留原始元数据
    )
    cleaned_documents.append(cleaned_doc)


#将清洗后的文档切分

text_splitter = RecursiveCharacterTextSplitter(
    # 设置分隔符，这里使用 ".!?" 来匹配句子结束符号
    separators=["\n\n", "\n", "(?<=\。！？ )", " ", ""],
    # 设置每个片段的最大长度 chunk_size 一般根据文档内容或大小来设置,
    chunk_size=50,
    # 设置每个片段之间的重叠字符数量   chunk_overlap = chunk_size * 20% 或 30%
    chunk_overlap=10,
    # 设置是否长内容优先
    length_function=len
)

chunks_list = text_splitter.split_documents(cleaned_documents)

chunks = [c for c in chunks_list if len(c.page_content)>30]  #过滤


#将切分后的文档保存到向量数据库中


db = Chroma.from_documents(
    documents=chunks,
    collection_name="car_info", #集合名称
    embedding=embedding_model,
    persist_directory="/teacher_data/wtb/教学文档/大模型教学文档/langchain_47/20261109RAG/chroma_db"# 储存的位置
)


query="谁是老师"
results = db.similarity_search_with_score(query,k=2)
print(results)



#使用已经保存到向量数据库中的数据进行查询

# db = Chroma(
#     collection_name="car_info",
#     embedding_function=embedding_model,
#     persist_directory="/teacher_data/wtb/教学文档/大模型教学文档/langchain_47/20261109RAG/chroma_db"
# )

# query="谁是老师"
# results = db.similarity_search(query,k=2)
# print(results)







