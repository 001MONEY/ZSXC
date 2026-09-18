"""
Seq2Seq 机器翻译的数据处理模块。

功能：
1. 读取 cmn.txt 中的中英平行语料；
2. 对中文和英文分别分词；
3. 建立词表；
4. 完成文本与编号之间的 encode/decode；
5. 将词表保存为 JSON，并从 JSON 恢复。
"""

from collections import Counter
import json
import logging
from pathlib import Path

import jieba
import nltk
from zhconv import zhconv

from config import Config


# 关闭 jieba 的调试日志，避免分词时反复打印初始化信息。
jieba.setLogLevel(logging.ERROR)

# TreebankWordTokenizer 不需要额外下载 NLTK 数据包。
__en_nltk = nltk.TreebankWordTokenizer()


def load_cmn_to_pairs(file_path):
    """读取中英平行语料，返回 ``[(中文, 英文), ...]``。"""
    pairs = []

    with open(file_path, "rt", encoding="UTF-8") as f:
        for line_number, line in enumerate(f, start=1):
            # cmn.txt 通常为：英文\t中文\t来源信息。
            fields = line.rstrip("\r\n").split("\t")

            # 跳过空行或缺少中英字段的异常行。
            if len(fields) < 2:
                continue

            en = fields[0].strip()
            zh = fields[1].strip()

            if not zh or not en:
                continue

            pairs.append((zh, en))

    return pairs


def zh_tokenizer(zh_text):
    """将中文统一为简体，然后使用 jieba 分词。"""
    simplified_text = zhconv.convert(zh_text.strip(), "zh-cn")
    return [token for token in jieba.lcut(simplified_text) if token.strip()]


def en_tokenizer(en_text):
    """使用 NLTK TreebankWordTokenizer 对英文分词。"""
    return [token for token in __en_nltk.tokenize(en_text.strip()) if token.strip()]


class Vocabulary:
    """维护 Token、编号及词频之间的映射。"""

    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = (
        "<PAD>",
        "<SOS>",
        "<EOS>",
        "<UNK>",
    )
    PAD_INDEX, SOS_INDEX, EOS_INDEX, UNK_INDEX = 0, 1, 2, 3
    __INIT_VOCAB = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    def __init__(self):
        self.vocab = list(self.__INIT_VOCAB)
        self.counter = Counter()
        self._rebuild_mappings()

    def _rebuild_mappings(self):
        """根据 vocab 同步生成正向和反向映射。"""
        self.word2index = {word: index for index, word in enumerate(self.vocab)}
        self.index2word = {index: word for index, word in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    def generate_vocab(self, pairs, text_index, tokenizer_fun, min_freq=1):
        """
        根据平行语料建立词表。

        Args:
            pairs: ``[(中文, 英文), ...]``。
            text_index: 0 表示中文，1 表示英文。
            tokenizer_fun: 对应语言的分词函数。
            min_freq: 只保留出现次数不小于该值的 Token。
        """
        if text_index not in (0, 1):
            raise ValueError("text_index 只能是 0（中文）或 1（英文）")
        if min_freq < 1:
            raise ValueError("min_freq 必须大于或等于 1")

        self.counter.clear()

        for pair in pairs:
            tokens = tokenizer_fun(pair[text_index])
            self.counter.update(tokens)

        # Counter 会保留 Token 第一次出现的顺序，方便复现实验结果。
        learned_tokens = [
            token
            for token, count in self.counter.items()
            if count >= min_freq and token not in self.__INIT_VOCAB
        ]
        self.vocab = list(self.__INIT_VOCAB) + learned_tokens
        self._rebuild_mappings()

        return self

    def encode(self, text, tokenizer_fun, add_sos=False, add_eos=True):
        """将一句文本转换成编号列表，未登录词映射为 ``<UNK>``。"""
        tokens = tokenizer_fun(text)
        indices = [
            self.word2index.get(token, self.UNK_INDEX)
            for token in tokens
        ]

        if add_sos:
            indices.insert(0, self.SOS_INDEX)
        if add_eos:
            indices.append(self.EOS_INDEX)

        return indices

    def decode(self, indices, stop_at_eos=True, skip_special_tokens=True):
        """将编号序列还原成 Token 列表。"""
        tokens = []
        special_tokens = {
            self.PAD_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN,
        }

        for index in indices:
            # 兼容 Python int、NumPy 标量和 PyTorch 标量张量。
            index = int(index)
            token = self.index2word.get(index, self.UNK_TOKEN)

            if token == self.EOS_TOKEN and stop_at_eos:
                break
            if skip_special_tokens and token in special_tokens:
                continue

            tokens.append(token)

        return tokens

    def save(self, file_path="word2index.json"):
        """把词表及词频保存为 UTF-8 JSON 文件。"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "word2index": self.word2index,
            "counter": dict(self.counter),
        }

        with open(file_path, "wt", encoding="UTF-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return file_path

    @classmethod
    def load(cls, file_path="word2index.json"):
        """从 JSON 恢复词表，并重建 vocab 和 index2word。"""
        with open(file_path, "rt", encoding="UTF-8") as f:
            payload = json.load(f)

        # 同时兼容 {"word2index": {...}} 和直接保存映射的旧格式。
        raw_mapping = payload.get("word2index", payload)
        word2index = {word: int(index) for word, index in raw_mapping.items()}

        if len(set(word2index.values())) != len(word2index):
            raise ValueError("词表中存在重复编号，无法恢复")

        expected_indices = set(range(len(word2index)))
        if set(word2index.values()) != expected_indices:
            raise ValueError("词表编号必须从 0 开始且保持连续")

        required_tokens = {
            cls.PAD_TOKEN: cls.PAD_INDEX,
            cls.SOS_TOKEN: cls.SOS_INDEX,
            cls.EOS_TOKEN: cls.EOS_INDEX,
            cls.UNK_TOKEN: cls.UNK_INDEX,
        }
        for token, expected_index in required_tokens.items():
            if word2index.get(token) != expected_index:
                raise ValueError(f"特殊 Token {token} 的编号应为 {expected_index}")

        vocab = cls()
        vocab.word2index = word2index
        vocab.index2word = {index: word for word, index in word2index.items()}
        vocab.vocab = [
            vocab.index2word[index]
            for index in range(len(vocab.index2word))
        ]
        vocab.counter = Counter(payload.get("counter", {}))

        return vocab


def build_vocabularies(pairs, min_freq=1):
    """分别建立中文源词表和英文目标词表。"""
    zh_vocab = Vocabulary().generate_vocab(
        pairs,
        text_index=0,
        tokenizer_fun=zh_tokenizer,
        min_freq=min_freq,
    )
    en_vocab = Vocabulary().generate_vocab(
        pairs,
        text_index=1,
        tokenizer_fun=en_tokenizer,
        min_freq=min_freq,
    )
    return zh_vocab, en_vocab


if __name__ == "__main__":
    sentence_pairs = load_cmn_to_pairs(Config.FILE_PATH)
    zh_vocab, en_vocab = build_vocabularies(sentence_pairs)

    # 中英文使用不同的词表，分别保存，避免映射关系相互覆盖。
    output_dir = Path(__file__).resolve().parent
    zh_vocab_path = zh_vocab.save(output_dir / "zh_word2index.json")
    en_vocab_path = en_vocab.save(output_dir / "en_word2index.json")

    sample_zh, sample_en = sentence_pairs[0]
    sample_zh_ids = zh_vocab.encode(sample_zh, zh_tokenizer)
    sample_en_ids = en_vocab.encode(
        sample_en,
        en_tokenizer,
        add_sos=True,
        add_eos=True,
    )

    print("第一组中英句子：", (sample_zh, sample_en))
    print("中文编码：", sample_zh_ids)
    print("中文解码：", zh_vocab.decode(sample_zh_ids))
    print("英文编码：", sample_en_ids)
    print("英文解码：", en_vocab.decode(sample_en_ids))
    print("中文词表大小：", len(zh_vocab))
    print("英文词表大小：", len(en_vocab))
    print("中文词表已保存：", zh_vocab_path)
    print("英文词表已保存：", en_vocab_path)
