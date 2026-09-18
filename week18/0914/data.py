"""
数据模块

中文分词： jieba
英文分词： NLTK
     pip install nltk

我们只会使用到简体中文，所以我们将语料库中的所有中文全部转成简体使用
    pip install zhconv
"""
from config import Config
from zhconv import zhconv
from collections import Counter
import jieba
import logging
import nltk
import json

# 设置jieba的日志级别
jieba.setLogLevel(logging.ERROR)

# 创建NLTK分词器对象
__en_nltk = nltk.TreebankWordTokenizer()


def load_cmn_to_pairs(file_path):
    """
    加载语料文件， 按照\t分割， 获取出中文和英文
    :param file_path: 文件路径
    :return:  返回成对的中文和英文
    """
    pairs = []
    with open(file_path, 'rt', encoding="UTF-8") as f:
        for line in f:
            words = line.split("\t")
            en = words[0]
            zh = words[1]
            pairs.append((zh, en))
    return pairs


def zh_tokenizer(zh_text):
    """
    中文分词
    :param zh_text: 传入的中文句子
    :return:
    """
    return jieba.lcut(zhconv.convert(zh_text, "zh-cn"))


def en_tokenizer(en_text):
    """
    英文分词
    :param en_text: 传入的英文句子
    :return:
    """
    return __en_nltk.tokenize(en_text)


# 词汇表类
class Vocabulary:
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
    PAD_INDEX, SOS_INDEX, EOS_INDEX, UNK_INDEX = 0, 1, 2, 3
    __INIT_VOCAB = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    def __init__(self):
        self.vocab = list(self.__INIT_VOCAB)
        self.word2index = {}
        self.index2word = {}
        self.counter = Counter()

    def generate_vocal(self, pairs, text_index, tokenizer_fun):
        # pair: ("静静的，别动。", "Be still.")
        for pair in pairs:
            # text_index=0, 表示中文
            # text_index=1, 表示英文
            # tokens=[静静, 的, 别动, 。]
            tokens = tokenizer_fun(pair[text_index])
            self.counter.update(tokens)

        # 构建词汇表vocab
        self.vocab += [token for token in self.counter]

        # 构建word2index和index2word
        self.word2index = {word: i for i, word in enumerate(self.vocab)}
        self.index2word = {i: word for i, word in enumerate(self.vocab)}

    def encode(self):
        pass

    def decode(self):
        pass

    def save(self):
        # 将word2index持久化成json文件
        with open("word2index.json", "wt", encoding="UTF-8") as f:
            json.dump({"word2index": self.word2index}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls):
        vocab = Vocabulary()
        with open("word2index.json", "rt", encoding="UTF-8") as f:
            vocab.word2index = json.load(f)["word2index"]


        vocab.vocab = []
        vocab.index2word = {}

        return vocab


if __name__ == '__main__':
    vocab = Vocabulary()
    pairs = load_cmn_to_pairs(Config.FILE_PATH)
    vocab.generate_vocal(pairs, 0, zh_tokenizer)
    vocab.load()
