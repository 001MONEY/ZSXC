#加载各种类型的数据 以pdf为例

# pip install pymupdf==1.25.5 -i https://pypi.tuna.tsinghua.edu.cn/simple

from langchain_community.document_loaders import PyMuPDFLoader

pdf_loader = PyMuPDFLoader("/teacher_data/wtb/教学文档/大模型教学文档/large_model/bigmodel/RAG/car_info.pdf")

documents = pdf_loader.load()  #按页来加载

# print(documents)

# print(len(documents))



# print(type(documents[0]))

print(documents[0])


# 加载word文档

'''
模式	粒度	适用场景
single	整个文档	文档较短，需整体处理
elements	段落/标题/列表等元素	需要精细结构化解析，保留语义边界
paged	按页	需要按页面区分内容

使用 elements 模式时，每个 Document 的 metadata 会包含额外信息：
{
    "source": "xxx.docx",
    "category": "Title",        # 元素类型
    "element_id": "abc123",     # 元素唯一ID
    "filename": "xxx.docx",
    "page_number": 1,
    ...
}

category 常见值包括：Title、NarrativeText、ListItem、Table、Header、Footer 等。
需要对category进行过滤
'''


# from langchain_community.document_loaders import UnstructuredWordDocumentLoader

# word_loader = UnstructuredWordDocumentLoader("/teacher_data/wtb/教学文档/大模型教学文档/langchain_47/20261109RAG/峨眉山讲解词.docx",
#                                             mode="elements")

# documents = word_loader.load()

# print(len(documents))

# print(type(documents[0]))

# print(documents[0])

