"""
数据模块：读取中英平行语料、分词、构建词表、生成 DataLoader。

中文分词：jieba（先用 zhconv 统一转成简体）
英文分词：NLTK 的 TreebankWordTokenizer
依赖安装：
    pip install nltk
    pip install zhconv
"""
import json
import logging
import random
from collections import Counter

import jieba
import nltk
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from zhconv import zhconv

from config import Config

# 词表文件保存在 raw/ 目录下，存在则直接加载，不用重新统计
ZH_VOCAB_PATH = Config.DATA_PATH / "zh.json"
EN_VOCAB_PATH = Config.DATA_PATH / "en.json"

# 设置 jieba 的日志级别
jieba.setLogLevel(logging.ERROR)

# NLTK 的 Treebank 分词器是无状态的，全局创建一个即可
_EN_TOKENIZER = nltk.TreebankWordTokenizer()


def load_cmn_to_pairs(file_path):
    """读取语料文件，每行形如 `英文\t中文`，返回 (中文, 英文) 句子对列表。"""
    pairs = []
    with open(file_path, "rt", encoding="UTF-8") as f:
        for line in f:
            en_text, zh_text = line.split("\t")[:2]
            pairs.append((zh_text, en_text))
    return pairs


def zh_tokenizer(zh_text):
    """中文分词：先转简体，再交给 jieba。"""
    return jieba.lcut(zhconv.convert(zh_text, "zh-cn"))


def en_tokenizer(en_text):
    """英文分词。"""
    return _EN_TOKENIZER.tokenize(en_text)


class Vocabulary:
    """词表：负责 Token 与编号的互相转换，并持久化成 json 文件。"""

    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
    PAD_INDEX, SOS_INDEX, EOS_INDEX, UNK_INDEX = 0, 1, 2, 3

    def __init__(self):
        # 前 4 个编号固定留给特殊 Token
        self.vocab = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        self.word2index = {}
        self.index2word = {}
        self.counter = Counter()

    def __len__(self):
        return len(self.vocab)

    def build_vocab(self, sentences, tokenizer_fun, save_path):
        """统计词频构建词表，并保存到 save_path。

        sentences:     句子列表（只传训练集，避免验证集信息泄漏）
        tokenizer_fun: 分词函数（中文 zh_tokenizer / 英文 en_tokenizer）
        """
        for sentence in sentences:
            self.counter.update(tokenizer_fun(sentence))

        # 按首次出现的顺序拼在特殊 Token 后面
        self.vocab += list(self.counter)

        self.word2index = {word: index for index, word in enumerate(self.vocab)}
        self.index2word = {index: word for index, word in enumerate(self.vocab)}
        self.save(save_path)

    def encode(self, tokens, add_sos=False, add_eos=True):
        """Token 列表 -> 编号列表，词表外的词映射为 <UNK>。

        add_sos / add_eos 控制是否在首尾插入 <SOS> / <EOS>。
        """
        indexes = [self.word2index.get(token, self.UNK_INDEX) for token in tokens]
        if add_sos:
            indexes = [self.SOS_INDEX] + indexes
        if add_eos:
            indexes = indexes + [self.EOS_INDEX]
        return indexes

    def decode(self, indexes, keep_special=True):
        """编号列表 -> Token 列表；keep_special=False 时跳过 <PAD>/<SOS>/<EOS>。"""
        tokens = []
        for index in indexes:
            token = self.index2word[index]
            if not keep_special and token in (self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN):
                continue
            tokens.append(token)
        return tokens

    def save(self, file_path):
        """把 word2index 保存成 json 文件。"""
        with open(file_path, "wt", encoding="UTF-8") as f:
            json.dump({"word2index": self.word2index}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path):
        """从 json 文件恢复 Vocabulary 对象。"""
        vocab = cls()
        with open(file_path, "rt", encoding="UTF-8") as f:
            vocab.word2index = json.load(f)["word2index"]
        vocab.index2word = {index: word for word, index in vocab.word2index.items()}
        vocab.vocab = [vocab.index2word[index] for index in range(len(vocab.word2index))]
        return vocab


class TranslationDataset(Dataset):
    """把 (中文, 英文) 句子对提前编码成编号张量，避免每个 epoch 重复分词。"""

    def __init__(self, pairs, zh_vocab, en_vocab):
        super().__init__()
        # source_tensors: Encoder 输入，中文内容 + <EOS>
        # target_tensors: Decoder 监督信号，<SOS> + 英文内容 + <EOS>
        self.source_tensors = []
        self.target_tensors = []

        for zh_text, en_text in pairs:
            zh_tokens = zh_tokenizer(zh_text)
            en_tokens = en_tokenizer(en_text)
            self.source_tensors.append(
                torch.tensor(zh_vocab.encode(zh_tokens, add_eos=True), dtype=torch.long)
            )
            self.target_tensors.append(
                torch.tensor(en_vocab.encode(en_tokens, add_sos=True, add_eos=True), dtype=torch.long)
            )

    def __len__(self):
        return len(self.source_tensors)

    def __getitem__(self, index):
        return self.source_tensors[index], self.target_tensors[index]


def make_collate_fn(pad_index):
    """生成 collate_fn：把同一批长短不一的句子补成相同长度。"""

    def collate_fn(batch):
        # DataLoader 只会传 batch，所以用闭包把 pad_index 固定进来
        source_sequences, target_sequences = zip(*batch)

        # 必须在补 PAD 之前记录真实长度，Encoder 靠它跳过 PAD
        src_lengths = torch.tensor(
            [len(sequence) for sequence in source_sequences],
            dtype=torch.long,
        )

        # 按当前 batch 中最长的句子补 PAD，而不是全量数据里的最长句
        source_padded = pad_sequence(
            source_sequences,
            batch_first=True,
            padding_value=pad_index,
        )
        target_padded = pad_sequence(
            target_sequences,
            batch_first=True,
            padding_value=pad_index,
        )
        return source_padded, target_padded, src_lengths

    return collate_fn


def create_dataloaders():
    """构建训练/验证 DataLoader 和两个词表。

    返回 (train_loader, val_loader, train_pairs, val_pairs, zh_vocab, en_vocab)。
    句子对也返回给调用方，方便训练中展示翻译示例、评估 BLEU。
    """
    # 1. 读取语料，打乱后按比例划分
    #    原始语料前短后长，不 shuffle 直接切会让验证集全是长句
    pairs = load_cmn_to_pairs(Config.FILE_PATH)
    random.Random(Config.SEED).shuffle(pairs)
    train_size = int(len(pairs) * Config.TRAIN_VAL_RATE)
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]

    # 2. 词表只根据训练集构建；文件已存在则直接加载
    if ZH_VOCAB_PATH.exists() and EN_VOCAB_PATH.exists():
        zh_vocab = Vocabulary.load(ZH_VOCAB_PATH)
        en_vocab = Vocabulary.load(EN_VOCAB_PATH)
    else:
        zh_vocab = Vocabulary()
        en_vocab = Vocabulary()
        zh_sentences = [zh_text for zh_text, _ in train_pairs]
        en_sentences = [en_text for _, en_text in train_pairs]
        zh_vocab.build_vocab(zh_sentences, zh_tokenizer, ZH_VOCAB_PATH)
        en_vocab.build_vocab(en_sentences, en_tokenizer, EN_VOCAB_PATH)

    # 3. 数据集和加载器
    train_dataset = TranslationDataset(train_pairs, zh_vocab, en_vocab)
    val_dataset = TranslationDataset(val_pairs, zh_vocab, en_vocab)

    collate_fn = make_collate_fn(zh_vocab.PAD_INDEX)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, train_pairs, val_pairs, zh_vocab, en_vocab


if __name__ == '__main__':
    train_loader, val_loader, train_pairs, val_pairs, zh_vocab, en_vocab = create_dataloaders()

    print("训练 / 验证句对数：", len(train_pairs), "/", len(val_pairs))
    print("中文词表大小：", len(zh_vocab))
    print("英文词表大小：", len(en_vocab))
    print("训练集批次数：", len(train_loader))
    print("验证集批次数：", len(val_loader))

    # 取一个 batch 检查形状和内容
    source, target, src_lengths = next(iter(train_loader))
    print("source：", tuple(source.shape), "target：", tuple(target.shape))
    print("src_lengths：", src_lengths.tolist())
    print("第 0 条中文：", zh_vocab.decode(source[0].tolist(), keep_special=False))
    print("第 0 条英文：", en_vocab.decode(target[0].tolist(), keep_special=False))

