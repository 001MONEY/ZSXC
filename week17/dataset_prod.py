"""工程版 CBOW 数据管道：持久化词表，并从大语料中流式生成样本。"""

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterator

import jieba
import torch
from torch.utils.data import IterableDataset, get_worker_info


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
_WORD_PATTERN = re.compile(r"\w")


def tokenize(line: str) -> list[str]:
    """中文分词，并丢掉纯标点、纯符号和纯空白 token。"""
    return [
        token
        for token in jieba.lcut(line)
        if _WORD_PATTERN.search(token)
    ]


def iter_corpus_lines(corpus_path: str | Path) -> Iterator[str]:
    """逐行读取 UTF-8 语料。"""
    path = Path(corpus_path)
    with path.open("rt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield line


class Vocabulary:
    """稳定、可保存和加载的词表。"""

    def __init__(self, tokens: list[str], counts: list[int]) -> None:
        if len(tokens) != len(counts):
            raise ValueError("tokens 和 counts 的长度必须一致")
        if len(tokens) < 2 or tokens[:2] != [PAD_TOKEN, UNK_TOKEN]:
            raise ValueError("词表前两个位置必须是 <PAD> 和 <UNK>")
        if len(tokens) != len(set(tokens)):
            raise ValueError("词表中不能包含重复 token")

        self.tokens = tokens
        self.counts = counts
        self.word_to_index = {
            token: index
            for index, token in enumerate(tokens)
        }

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def pad_index(self) -> int:
        return self.word_to_index[PAD_TOKEN]

    @property
    def unk_index(self) -> int:
        return self.word_to_index[UNK_TOKEN]

    def encode(self, tokens: list[str]) -> list[int]:
        return [
            self.word_to_index.get(token, self.unk_index)
            for token in tokens
        ]

    def save(self, file_path: str | Path) -> None:
        payload = {
            "tokens": self.tokens,
            "counts": self.counts,
        }
        Path(file_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, file_path: str | Path) -> "Vocabulary":
        payload = json.loads(
            Path(file_path).read_text(encoding="utf-8")
        )
        return cls(
            tokens=payload["tokens"],
            counts=payload["counts"],
        )

    @classmethod
    def build(
        cls,
        corpus_path: str | Path,
        min_freq: int = 5,
        max_vocab_size: int = 50_000,
    ) -> "Vocabulary":
        if min_freq < 1:
            raise ValueError("min_freq 必须大于等于 1")
        if max_vocab_size < 3:
            raise ValueError("max_vocab_size 至少为 3")

        counter = Counter()
        for line in iter_corpus_lines(corpus_path):
            counter.update(tokenize(line))

        special_tokens = {PAD_TOKEN, UNK_TOKEN}
        valid_items = [
            (token, count)
            for token, count in counter.most_common()
            if count >= min_freq and token not in special_tokens
        ][: max_vocab_size - 2]

        valid_token_set = {token for token, _ in valid_items}
        unknown_count = sum(
            count
            for token, count in counter.items()
            if token not in valid_token_set
        )

        tokens = [PAD_TOKEN, UNK_TOKEN] + [
            token for token, _ in valid_items
        ]
        counts = [0, unknown_count] + [
            count for _, count in valid_items
        ]

        return cls(tokens=tokens, counts=counts)


class CBOWStreamingDataset(IterableDataset):
    """不缓存全部 context/target，训练时按行生成定长 token ID 样本。"""

    def __init__(
        self,
        corpus_path: str | Path,
        vocabulary: Vocabulary,
        window_size: int = 2,
        shuffle_buffer_size: int = 10_000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if window_size < 1:
            raise ValueError("window_size 必须大于等于 1")
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size 不能小于 0")

        self.corpus_path = Path(corpus_path)
        if not self.corpus_path.is_file():
            raise FileNotFoundError(f"找不到语料文件：{self.corpus_path}")

        self.vocabulary = vocabulary
        self.window_size = window_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """让每轮训练使用不同但可复现的缓冲区随机顺序。"""
        self.epoch = epoch

    def _generate_samples(
        self,
        worker_id: int,
        num_workers: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        pad_index = self.vocabulary.pad_index
        unk_index = self.vocabulary.unk_index
        window_size = self.window_size

        for line_number, line in enumerate(
            iter_corpus_lines(self.corpus_path)
        ):
            # 多进程 DataLoader 下避免各 worker 产生重复样本。
            if line_number % num_workers != worker_id:
                continue

            token_ids = self.vocabulary.encode(tokenize(line))
            if not token_ids:
                continue

            padded = (
                [pad_index] * window_size
                + token_ids
                + [pad_index] * window_size
            )

            for index, target in enumerate(token_ids):
                # 被折叠的低频词可以作为上下文，但不作为预测目标。
                if target == unk_index:
                    continue

                center = index + window_size
                context = (
                    padded[center - window_size:center]
                    + padded[center + 1:center + window_size + 1]
                )

                yield (
                    torch.tensor(context, dtype=torch.long),
                    torch.tensor(target, dtype=torch.long),
                )

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        samples = self._generate_samples(worker_id, num_workers)
        if self.shuffle_buffer_size <= 1:
            yield from samples
            return

        # IterableDataset 不能使用 DataLoader(shuffle=True)，因此使用
        # 有界缓冲区近似打乱，内存占用不随整个语料规模增长。
        rng = random.Random(
            self.seed + self.epoch * 1_000_003 + worker_id
        )
        buffer: list[tuple[torch.Tensor, torch.Tensor]] = []

        for sample in samples:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(sample)
                continue

            selected = rng.randrange(len(buffer))
            yield buffer[selected]
            buffer[selected] = sample

        rng.shuffle(buffer)
        yield from buffer
