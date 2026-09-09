"""工程版 CBOW 训练入口：配置、负采样、检查点、恢复训练和导出。"""

import argparse
import logging
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from dataset_prod import CBOWStreamingDataset, Vocabulary
from model_prod import CBOWNegativeSampling


LOGGER = logging.getLogger("cbow")
BASE_DIR = Path(__file__).resolve().parent


@dataclass
class TrainConfig:
    corpus_path: Path = BASE_DIR / "sentence.txt"
    output_dir: Path = BASE_DIR / "artifacts"
    min_freq: int = 1
    max_vocab_size: int = 50_000
    window_size: int = 2
    embedding_dim: int = 32
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.003
    num_negative: int = 5
    shuffle_buffer_size: int = 10_000
    num_workers: int = 0
    seed: int = 42
    resume: bool = False
    rebuild_vocab: bool = False


class NegativeSampler:
    """按 unigram^0.75 分布采样负例。"""

    def __init__(self, vocabulary: Vocabulary) -> None:
        weights = torch.tensor(
            vocabulary.counts,
            dtype=torch.float,
        ).pow(0.75)
        weights[vocabulary.pad_index] = 0
        weights[vocabulary.unk_index] = 0

        if torch.count_nonzero(weights) < 2:
            raise ValueError("有效词太少，无法进行负采样")

        self.weights = weights

    def sample(
        self,
        target_ids: torch.Tensor,
        num_negative: int,
        device: torch.device,
    ) -> torch.Tensor:
        targets_cpu = target_ids.detach().cpu()
        negatives = torch.multinomial(
            self.weights,
            num_samples=targets_cpu.numel() * num_negative,
            replacement=True,
        ).view(targets_cpu.size(0), num_negative)

        # 避免把当前正样本同时当成负样本。
        conflicts = negatives.eq(targets_cpu.unsqueeze(1))
        while conflicts.any():
            negatives[conflicts] = torch.multinomial(
                self.weights,
                num_samples=int(conflicts.sum().item()),
                replacement=True,
            )
            conflicts = negatives.eq(targets_cpu.unsqueeze(1))

        return negatives.to(device, non_blocking=True)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload: dict, file_path: Path) -> None:
    """先写临时文件再替换，降低中途退出导致检查点损坏的风险。"""
    temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, file_path)


def config_to_dict(config: TrainConfig) -> dict:
    payload = asdict(config)
    payload["corpus_path"] = str(config.corpus_path)
    payload["output_dir"] = str(config.output_dir)
    return payload


def save_word_vectors(
    model: CBOWNegativeSampling,
    vocabulary: Vocabulary,
    file_path: Path,
) -> None:
    vectors = model.get_token_embeddings().detach().cpu()
    valid_indices = [
        index
        for index in range(vocabulary.size)
        if index != vocabulary.pad_index
    ]

    with file_path.open("wt", encoding="utf-8") as file:
        file.write(f"{len(valid_indices)} {vectors.size(1)}\n")
        for index in valid_indices:
            values = " ".join(
                f"{value:.6f}"
                for value in vectors[index].tolist()
            )
            file.write(f"{vocabulary.tokens[index]} {values}\n")


def train(config: TrainConfig) -> tuple[CBOWNegativeSampling, Vocabulary]:
    set_random_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = config.output_dir / "vocab.json"
    latest_path = config.output_dir / "latest.pt"
    best_path = config.output_dir / "best.pt"
    vectors_path = config.output_dir / "word2vec.txt"

    if vocab_path.exists() and not config.rebuild_vocab:
        vocabulary = Vocabulary.load(vocab_path)
        LOGGER.info("加载已有词表：%s 个词", vocabulary.size)
    else:
        vocabulary = Vocabulary.build(
            corpus_path=config.corpus_path,
            min_freq=config.min_freq,
            max_vocab_size=config.max_vocab_size,
        )
        vocabulary.save(vocab_path)
        LOGGER.info("建立并保存词表：%s 个词", vocabulary.size)

    dataset = CBOWStreamingDataset(
        corpus_path=config.corpus_path,
        vocabulary=vocabulary,
        window_size=config.window_size,
        shuffle_buffer_size=config.shuffle_buffer_size,
        seed=config.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CBOWNegativeSampling(
        vocab_size=vocabulary.size,
        embedding_dim=config.embedding_dim,
        pad_index=vocabulary.pad_index,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )
    negative_sampler = NegativeSampler(vocabulary)

    start_epoch = 0
    best_loss = float("inf")

    if config.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        LOGGER.info("从第 %s 轮继续训练", start_epoch + 1)

    for epoch in range(start_epoch, config.epochs):
        dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_samples = 0

        for context_ids, target_ids in dataloader:
            context_ids = context_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            negative_ids = negative_sampler.sample(
                target_ids=target_ids,
                num_negative=config.num_negative,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            loss = model(context_ids, target_ids, negative_ids)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            current_batch_size = target_ids.size(0)
            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

        if total_samples == 0:
            raise RuntimeError(
                "没有生成训练样本，请检查语料或降低 min_freq。"
            )

        average_loss = total_loss / total_samples
        checkpoint = {
            "epoch": epoch,
            "best_loss": min(best_loss, average_loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config_to_dict(config),
        }
        atomic_torch_save(checkpoint, latest_path)

        if average_loss < best_loss:
            best_loss = average_loss
            checkpoint["best_loss"] = best_loss
            atomic_torch_save(checkpoint, best_path)

        LOGGER.info(
            "Epoch %d/%d | samples=%d | loss=%.6f | best=%.6f",
            epoch + 1,
            config.epochs,
            total_samples,
            average_loss,
            best_loss,
        )

    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    save_word_vectors(model, vocabulary, vectors_path)

    LOGGER.info("最优模型：%s", best_path)
    LOGGER.info("词向量：%s", vectors_path)
    return model, vocabulary


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="训练工程版 CBOW 词向量")
    parser.add_argument("--corpus", type=Path, default=BASE_DIR / "sentence.txt")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "artifacts")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--num-negative", type=int, default=5)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-vocab", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_negative=args.num_negative,
        shuffle_buffer_size=args.shuffle_buffer_size,
        num_workers=args.num_workers,
        seed=args.seed,
        resume=args.resume,
        rebuild_vocab=args.rebuild_vocab,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    train(parse_args())
