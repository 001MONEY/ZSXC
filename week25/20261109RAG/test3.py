#文本分割器



    
#加载文档
from  langchain_community.document_loaders import PyMuPDFLoader
pdf_loader = PyMuPDFLoader("/teacher_data/wtb/教学文档/大模型教学文档/large_model/bigmodel/RAG/car_info.pdf")
documents = pdf_loader.load()



#清洗文档


import re 


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





from langchain_text_splitters import RecursiveCharacterTextSplitter
# 定义文本切割器,按句子进行分割
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

# # 对文本进行分割
# text =""" 
# LangChain: 一个让你的LLM变得更强大的开源框架。
# LangChain 就是一个 LLM 编程框架，你想开发一个基于LLM应用，需要什么组件它都有，
# 直接使用就行；甚至针对常规的应用流程，它利用链(LangChain中Chain的由来)这个概念已经内置标准化方案了。
# 下面我们从新兴的大语言模型（LLM）技术栈的角度来看看为何它的理念这么受欢迎。
# """
# texts = text_splitter.split_text(text)
# print(texts)




from langchain_core.documents import Document


cleaned_documents = []  #清洗后的文档

for doc in documents:

    cleaned_content = clean_pdf_text_re(doc.page_content)  #纯文

    cleaned_doc = Document(
        page_content=cleaned_content,
        metadata=doc.metadata  #保留原始元数据
    )
    cleaned_documents.append(cleaned_doc)



chunks_list = text_splitter.split_documents(cleaned_documents)

chunks = [c for c in chunks_list if len(c.page_content)>30]

print(len(chunks))

print(chunks[0])




