#数据清洗 使用正则表达式

from langchain_community.document_loaders import PyMuPDFLoader
import re

pdf_loader = PyMuPDFLoader("/teacher_data/wtb/教学文档/大模型教学文档/large_model/bigmodel/RAG/car_info.pdf")

documents = pdf_loader.load()


# print(documents[0].page_content)


# 2. 清洗文本（在分割前）  基于正则清洗
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



# 数据清洗基于 jionlp

# pip install jionlp

import jionlp as jio


# text = "请拨打电话13800138000，或400-123-4567"

# print(jio.clean_text(text))

# print(jio.remove_phone_number(text))

# print(jio.remove_email(text))

# print(jio.remove_url(text))

def clean_pdf_text_jio(text: str) -> str:
    return jio.clean_text(text)




from langchain_core.documents import Document


cleaned_documents = []  #清洗后的文档

for doc in documents:

    cleaned_content = clean_pdf_text_re(doc.page_content)  #纯文本

    cleaned_content = clean_pdf_text_jio(cleaned_content)  #纯文本

    cleaned_doc = Document(
        page_content=cleaned_content,
        metadata=doc.metadata  #保留原始元数据
    )
    cleaned_documents.append(cleaned_doc)

print(type(cleaned_documents[0]))
print(cleaned_documents[0])


