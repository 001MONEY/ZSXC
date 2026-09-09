"""CBOW 数据集：读取语料、建立词表并生成 One-Hot 上下文样本。"""

import re
from collections import Counter
from pathlib import Path

import jieba
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CBOWDataset(Dataset):
    """教学版 CBOW Dataset。

    处理流程：逐行读取 -> 分词 -> 统计词频 -> 建立词表 ->
    Padding + 滑动窗口 -> 在取样时生成 One-Hot。
    """

    def __init__(
        self,
        root: str | Path,
        window_size: int = 2,
        min_freq: int = 2,
        max_vocab_size: int = 50_000,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size 必须大于等于 1")
        if min_freq < 1:
            raise ValueError("min_freq 必须大于等于 1")
        if max_vocab_size < 2:
            raise ValueError("max_vocab_size 至少为 2，需要容纳 <PAD> 和 <UNK>")

        self.root = Path(root)
        if not self.root.is_file():
            raise FileNotFoundError(f"找不到语料文件：{self.root}")

        self.window_size = window_size
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        self.create_vocab_index()
        self.create_context()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _load_text(self):
        """逐行读取，避免一次性把整个语料文件读入内存。"""
        with self.root.open("rt", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    yield line

    @staticmethod
    def _tokenize(line: str) -> list[str]:
        """中文分词，并丢掉纯标点、纯符号和纯空白 token。"""
        return [
            token
            for token in jieba.lcut(line)
            if re.search(r"\w", token)
        ]

    def create_vocab_index(self) -> None:
        """统计词频，过滤低频词，建立 word/index 双向映射。"""
        counter = Counter()

        # 第一遍扫描：只统计词频。
        for line in self._load_text():
            counter.update(self._tokenize(line))

        special_tokens = ["<PAD>", "<UNK>"]
        available_size = self.max_vocab_size - len(special_tokens)

        # 优先保留高频词，低频词不进入词表。
        valid_tokens = [
            token
            for token, count in counter.most_common()
            if count >= self.min_freq and token not in special_tokens
        ][:available_size]

        self.vocab = special_tokens + valid_tokens
        self.word2index = {
            word: index
            for index, word in enumerate(self.vocab)
        }
        self.index2word = {
            index: word
            for index, word in enumerate(self.vocab)
        }

    def create_context(self) -> None:
        """用 Padding + 滑动窗口生成 CBOW 的 context/target。"""
        self.contexts: list[list[int]] = []
        self.targets: list[int] = []

        pad_index = self.word2index["<PAD>"]
        unk_index = self.word2index["<UNK>"]
        window_size = self.window_size

        # 第二遍扫描：把每句话转换成训练样本。
        for line in self._load_text():
            sentence = self._tokenize(line)
            token_ids = [
                self.word2index.get(token, unk_index)
                for token in sentence
            ]

            if not token_ids:
                continue

            padded_token_ids = (
                [pad_index] * window_size
                + token_ids
                + [pad_index] * window_size
            )

            for i, target in enumerate(token_ids):
                # 低频词折叠后彼此无法区分，UNK 只作上下文，不作中心词。
                if target == unk_index:
                    continue

                center_index = i + window_size
                left_context = padded_token_ids[
                    center_index - window_size:center_index
                ]
                right_context = padded_token_ids[
                    center_index + 1:center_index + window_size + 1
                ]

                self.contexts.append(left_context + right_context)
                self.targets.append(target)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按需生成 One-Hot，避免长期保存巨大的 One-Hot 矩阵。"""
        context_ids = torch.tensor(
            self.contexts[index],
            dtype=torch.long,
        )
        target = torch.tensor(
            self.targets[index],
            dtype=torch.long,
        )

        context_one_hot = F.one_hot(
            context_ids,
            num_classes=self.vocab_size,
        ).float()

        # PAD 的普通 One-Hot 在第 0 位为 1，这里把整行清零。
        pad_index = self.word2index["<PAD>"]
        pad_mask = (context_ids != pad_index).unsqueeze(1)
        context_one_hot = context_one_hot * pad_mask

        # context_one_hot: [2 * window_size, vocab_size]
        # target: 中心词类别编号，可直接交给 CrossEntropyLoss。
        return context_one_hot, target


# 说明：这是便于理解完整流程的教学版。超大语料应使用 IterableDataset
# 流式生成样本；大词表正式训练通常直接传 token ID 给 nn.Embedding。
