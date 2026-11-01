#按需获取样例
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

from langchain_ollama import OllamaEmbeddings


import numpy as np


embedding1 = OllamaEmbeddings(model="qwen3-embedding:0.6b",base_url="http://192.168.1.105:11434")
embedding2 = OllamaEmbeddings(model="bge-large",base_url="http://192.168.1.105:11434")

#不是随便选的 基准测试 准确性


examples = [
    {"question": "你好吗?", "answer": "师哥，我很好"},
    {"question": "今天天气怎么样?", "answer": "师哥，今天晴天，适合出门"},
    {"question": "Python怎么写for循环?", "answer": "师哥，用 for i in range(n): 即可"},
    {"question": "红烧肉怎么做?", "answer": "师哥，五花肉焯水后加酱油慢炖"},
    {"question": "北京有什么好玩的?", "answer": "师哥，故宫和长城都很值得去"},
    {"question": "25乘以4等于多少?", "answer": "师哥，等于100"},
    {"question": "篮球怎么投三分?", "answer": "师哥，手肘对着篮筐，出手时手腕下压"},
    {"question": "最近有什么好看的电影?", "answer": "师哥，可以看看科幻片和悬疑片"},
    {"question": "失眠了怎么办?", "answer": "师哥，睡前少看手机，固定作息"},
    {"question": "秦始皇是谁?", "answer": "师哥，他是秦朝的开国皇帝"},
]


#分别使用不用的embedding模型计算不用大的相似度 然后求均值


def get_similarity(embedding,examples):
    similarities = []
    for example in examples:
        similarity = (np.array(embedding.embed_query(example["question"])) @ np.array(embedding.embed_query(example["answer"]))) / (np.linalg.norm(np.array(embedding.embed_query(example["question"]))) * np.linalg.norm(np.array(embedding.embed_query(example["answer"]))))
        similarities.append(similarity)
    return np.mean(similarities)


res1 = get_similarity(embedding1,examples)
print(res1)
print("--------------------------------")

res2 = get_similarity(embedding2,examples)
print(res2)

