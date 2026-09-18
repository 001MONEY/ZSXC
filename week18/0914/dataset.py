"""Seq2Seq 机器翻译的 Dataset 与 DataLoader。"""

import json
from pathlib import Path
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config import Config
from data_process import (
    Vocabulary,
    build_vocabularies,
    en_tokenizer,
    load_cmn_to_pairs,
    zh_tokenizer,
)


class TranslationDataset(Dataset):
    """将中英句子提前编码，避免每个 epoch 重复分词。"""

    def __init__(
        self,
        pairs,
        zh_vocab,
        en_vocab,
        max_src_length=30,
        max_tgt_length=30,
    ):
        self.samples = []
        self.filtered_count = 0

        for zh_text, en_text in pairs:
            # Encoder 输入：中文内容 + <EOS>。
            src_ids = zh_vocab.encode(
                zh_text,
                tokenizer_fun=zh_tokenizer,
                add_sos=False,
                add_eos=True,
            )

            # Decoder 完整序列：<SOS> + 英文内容 + <EOS>。
            target_full_ids = en_vocab.encode(
                en_text,
                tokenizer_fun=en_tokenizer,
                add_sos=True,
                add_eos=True,
            )

            # 超长句直接丢弃，不截断，避免破坏完整翻译和 <EOS>。
            if len(src_ids) > max_src_length:
                self.filtered_count += 1
                continue
            if len(target_full_ids) > max_tgt_length:
                self.filtered_count += 1
                continue

            # 只保存两份完整数据；后面通过切片构造教师强制输入和标签。
            self.samples.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(target_full_ids, dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        src_ids, target_full_ids = self.samples[index]

        return {
            "src": src_ids,
            "decoder_input": target_full_ids[:-1],
            "target": target_full_ids[1:],
        }


def translation_collate_fn(batch):
    """将不同长度的样本补 PAD，组成一个 Batch。"""
    src_sequences = [sample["src"] for sample in batch]
    decoder_sequences = [sample["decoder_input"] for sample in batch]
    target_sequences = [sample["target"] for sample in batch]

    # 必须在补 PAD 之前记录真实长度。
    src_lengths = torch.tensor(
        [len(sequence) for sequence in src_sequences],
        dtype=torch.long,
    )

    src_batch = pad_sequence(
        src_sequences,
        batch_first=True,
        padding_value=Vocabulary.PAD_INDEX,
    )
    decoder_batch = pad_sequence(
        decoder_sequences,
        batch_first=True,
        padding_value=Vocabulary.PAD_INDEX,
    )
    target_batch = pad_sequence(
        target_sequences,
        batch_first=True,
        padding_value=Vocabulary.PAD_INDEX,
    )

    return {
        "src": src_batch,
        "src_lengths": src_lengths,
        "decoder_input": decoder_batch,
        "target": target_batch,
    }


def load_or_create_split(
    pairs,
    save_path="train_val_split.json",
    val_ratio=0.1,
    seed=42,
):
    """读取已有划分；首次运行时随机划分并保存索引。"""
    save_path = Path(save_path)

    if save_path.exists():
        with open(save_path, "rt", encoding="UTF-8") as f:
            split = json.load(f)

        train_indices = split["train_indices"]
        val_indices = split["val_indices"]
    else:
        indices = list(range(len(pairs)))
        random.Random(seed).shuffle(indices)

        val_size = max(1, int(len(indices) * val_ratio))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        with open(save_path, "wt", encoding="UTF-8") as f:
            json.dump(
                {
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    train_pairs = [pairs[index] for index in train_indices]
    val_pairs = [pairs[index] for index in val_indices]
    return train_pairs, val_pairs


def create_dataloader(
    pairs,
    zh_vocab,
    en_vocab,
    batch_size=32,
    shuffle=True,
):
    """根据句子对创建 Dataset 和 DataLoader。"""
    dataset = TranslationDataset(pairs, zh_vocab, en_vocab)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=translation_collate_fn,
    )
    return dataset, dataloader


def create_train_val_dataloaders(
    train_pairs,
    val_pairs,
    zh_vocab,
    en_vocab,
    batch_size=32,
):
    """分别创建训练集和验证集的 DataLoader。"""
    train_dataset, train_loader = create_dataloader(
        train_pairs,
        zh_vocab,
        en_vocab,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataset, val_loader = create_dataloader(
        val_pairs,
        zh_vocab,
        en_vocab,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_dataset, val_dataset, train_loader, val_loader


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    # 1. 读取语料并固定 train/val 划分。
    pairs = load_cmn_to_pairs(Config.FILE_PATH)
    train_pairs, val_pairs = load_or_create_split(
        pairs,
        save_path=base_dir / "train_val_split.json",
        val_ratio=0.1,
        seed=42,
    )

    # 2. 只用训练集构建并保存词表。
    zh_vocab, en_vocab = build_vocabularies(train_pairs)
    zh_vocab.save(base_dir / "zh_train_word2index.json")
    en_vocab.save(base_dir / "en_train_word2index.json")

    # 3. 构建训练集、验证集和 DataLoader。
    train_dataset, val_dataset, train_loader, val_loader = (
        create_train_val_dataloaders(
            train_pairs,
            val_pairs,
            zh_vocab,
            en_vocab,
            batch_size=32,
        )
    )

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("全部样本数：", len(pairs))
    print("训练集样本数：", len(train_dataset))
    print("验证集样本数：", len(val_dataset))
    print("过滤超长句：", train_dataset.filtered_count + val_dataset.filtered_count)
    print("中文训练词表大小：", len(zh_vocab))
    print("英文训练词表大小：", len(en_vocab))
    print("train src shape：", tuple(train_batch["src"].shape))
    print("train target shape：", tuple(train_batch["target"].shape))
    print("val src shape：", tuple(val_batch["src"].shape))
    print("val target shape：", tuple(val_batch["target"].shape))
