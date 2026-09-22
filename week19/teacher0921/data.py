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
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import jieba
import logging
import nltk
import json
import torch
import random
import os

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

    def __len__(self):
        return len(self.vocab)

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

        self.__save("en.json" if text_index else "zh.json")

    def tokens2index(self, tokens, add_sos=False, add_eos=True):
        """
        传入seq2seq的Encoder和Decoder的数据
        :param tokens: 分词后的token
        :param add_sos: 是否添加SOS的索引
        :param add_eos: 是否添加EOS的索引
        :return:
        """
        indexes = [self.word2index.get(token, self.UNK_INDEX) for token in tokens]
        if add_sos:
            indexes = [self.SOS_INDEX] + indexes
        if add_eos:
            indexes = indexes + [self.EOS_INDEX]
        return indexes

    def index2tokens(self, indexes, is_special=True):
        """
        将id转成token
          解码器就可以使用：解码器预测生成的结果是ids， 调用这个方法就将ids转成token
        :param indexes: 预测生成的结果是ids
        :param is_special: True: 显示出SOS, EOS
        :return:
        """
        tokens = []
        for index in indexes:
            token = self.index2word[index]
            if not is_special and token in [self.SOS_TOKEN, self.EOS_TOKEN]:
                continue
            tokens.append(token)
        return tokens

    def __save(self, file_name):
        # 将word2index持久化成json文件
        with open(Config.DATA_PATH / file_name, "wt", encoding="UTF-8") as f:
            json.dump({"word2index": self.word2index}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_name):
        """
        加载保存的word2index.json文件
        :return: Vocabulary对象
        """
        vocab = cls()
        with open(Config.DATA_PATH / file_name, "rt", encoding="UTF-8") as f:
            vocab.word2index = json.load(f)["word2index"]
            vocab.index2word = {i:w for w, i in vocab.word2index.items()}
            vocab.vocab = [vocab.index2word[i] for i in range(len(vocab.word2index))]
        return vocab


class TranslationDataset(Dataset):

    def __init__(self, pairs, zh_vocab_obj:Vocabulary, en_vocab_obj:Vocabulary):
        super().__init__()
        # source_tensers存储输入模型的 中文token对应的id
        # target_tensers存储真实的结果 英文token对应的id
        self.source_tensers = []
        self.target_tensers = []

        for pair in pairs:
            # 获取中文，分词， 得到对应的id
            zh_tokens = zh_tokenizer(pair[0])
            en_tokens = en_tokenizer(pair[1])
            self.source_tensers.append(torch.tensor(zh_vocab_obj.tokens2index(zh_tokens, add_eos=True)))
            self.target_tensers.append(torch.tensor(en_vocab_obj.tokens2index(en_tokens, add_sos=True, add_eos=True)))


    def __len__(self):
        return len(self.source_tensers)

    def __getitem__(self, idx):
        return self.source_tensers[idx], self.target_tensers[idx]

# 闭包
def __make_collate_fn(pad_idx):
    # 该函数只能定义一个参数
    def collate_fn(batch):
        # batch中每条句子的长度不一样， 所以需要先统一长度再堆叠
        # 按照当前批次中最长的句子长度进行填充
        # 获取bacth中最长句子的长度
        source_tensers, target_tensers = zip(*batch)
        src_length = [len(src) for src in source_tensers]
        source_padded = pad_sequence(source_tensers, batch_first=True, padding_value=pad_idx)
        target_padded = pad_sequence(target_tensers, batch_first=True, padding_value=pad_idx)
        return source_padded, target_padded, src_length
    return collate_fn


def generate_dataloader():
    """
    生成训练集和验证集的数据加载器
    :return:
    """
    # 加载语料文件
    pairs = load_cmn_to_pairs(Config.FILE_PATH)
    # 划分训练集和验证集
    # 原始数据前面全是短句，后面全是长句， 所以需要将数据打乱在划分
    random.seed(Config.SEED) # 设置随机数的种子， 为了效果的复现
    random.shuffle(pairs)
    split_len = int(len(pairs) * Config.TRAIN_VAL_RATE)
    train_pairs = pairs[:split_len]
    val_pairs = pairs[split_len:]

    # zh.json和en.json存在的话就直接加载文件得到Vocabulary对象；否则才直接创建Vocabulary对象，保存文件
    zh_json_path = Config.DATA_PATH / "zh.json"
    en_json_path = Config.DATA_PATH / "en.json"
    if os.path.exists(zh_json_path) and os.path.exists(en_json_path):
        # 文件存在
        zh_vocab = Vocabulary.load(zh_json_path)
        en_vocab = Vocabulary.load(en_json_path)
    else:
        # 文件不存在
        zh_vocab = Vocabulary()
        en_vocab = Vocabulary()
        # 将训练集的数据保存成json即可，验证集无需保存
        zh_vocab.generate_vocal(train_pairs, 0, zh_tokenizer)
        en_vocab.generate_vocal(train_pairs, 1, en_tokenizer)

    # 训练数据集
    train_dataset = TranslationDataset(train_pairs, zh_vocab, en_vocab)
    # 验证数据集
    val_dataset = TranslationDataset(val_pairs, zh_vocab, en_vocab)

    # 加载器
    train_loader = DataLoader(dataset=train_dataset,batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=__make_collate_fn(zh_vocab.PAD_INDEX))
    val_loader = DataLoader(dataset=val_dataset,batch_size=Config.BATCH_SIZE, collate_fn=__make_collate_fn(zh_vocab.PAD_INDEX))

    return train_loader, val_loader, train_pairs, val_pairs, zh_vocab, en_vocab




if __name__ == '__main__':
    train_loader, _, _, _, _, _ = generate_dataloader()
    for src, tgt, src_lengths in train_loader:
        print("src:", tuple(src.shape))
        print("tgt:", tuple(tgt.shape))
        print("src_lengths:", len(src_lengths))
        break

