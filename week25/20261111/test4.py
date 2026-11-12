#BM25算法的使用

from langchain_community.retrievers import BM25Retriever

import jieba

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyPDFLoader("/teacher_data/wtb/教学文档/大模型教学文档/large_model/bigmodel/RAG/car_info.pdf")

docs = loader.load()

#切片处理
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=70,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
chunks = text_splitter.split_documents(docs)



#BM25算法

#定义一个中文分词函数


def chinese_tokenize(text):

    return [word for word in jieba.cut(text) if word.strip()]


bm25_retriever = BM25Retriever.from_documents(  
    documents=chunks,
    preprocess_func=chinese_tokenize
)


result = bm25_retriever.invoke("车顶的行李架载重多少")
print(result)

