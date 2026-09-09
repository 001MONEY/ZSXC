"""训练 CBOW，并保存最优模型和文本格式词向量。"""

import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CBOWDataset
from model import CBOWModel


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = BASE_DIR / "sentence.txt"


def save_word_vectors(
    model: CBOWModel,
    dataset: CBOWDataset,
    file_path: str | Path,
) -> None:
    """把词向量保存为常见的 Word2Vec 文本格式。"""
    token_embeddings = (
        model.get_token_embeddings()
        .detach()
        .cpu()
        .numpy()
    )

    pad_index = dataset.word2index["<PAD>"]
    valid_indices = [
        index
        for index in range(dataset.vocab_size)
        if index != pad_index
    ]
    embedding_dim = token_embeddings.shape[1]

    with Path(file_path).open("wt", encoding="utf-8") as file:
        file.write(f"{len(valid_indices)} {embedding_dim}\n")

        for index in valid_indices:
            token = dataset.index2word[index]
            vector_text = " ".join(
                f"{value:.6f}"
                for value in token_embeddings[index]
            )
            file.write(f"{token} {vector_text}\n")


def train(
    corpus_path: str | Path = DEFAULT_CORPUS,
    window_size: int = 2,
    min_freq: int = 1,
    max_vocab_size: int = 50_000,
    embedding_dim: int = 32,
    batch_size: int = 4,
    learning_rate: float = 0.01,
    epochs: int = 300,
) -> tuple[CBOWModel, CBOWDataset]:
    """完成数据加载、模型训练和最优参数保存。"""
    torch.manual_seed(42)

    dataset = CBOWDataset(
        root=corpus_path,
        window_size=window_size,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
    )
    if len(dataset) == 0:
        raise ValueError(
            "没有生成训练样本，请检查语料，或适当降低 min_freq。"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = CBOWModel(
        vocab_size=dataset.vocab_size,
        embedding_dim=embedding_dim,
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for context_one_hot, targets in dataloader:
            optimizer.zero_grad()

            logits = model(context_one_hot)
            loss = loss_fn(logits, targets)

            loss.backward()
            optimizer.step()

            current_batch_size = targets.size(0)
            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

        average_loss = total_loss / total_samples

        if average_loss < best_loss:
            best_loss = average_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(
                f"Epoch [{epoch + 1:3d}/{epochs}], "
                f"Loss: {average_loss:.6f}"
            )

    if best_state is None:
        raise RuntimeError("训练未产生有效的模型参数。")

    model.load_state_dict(best_state)

    checkpoint_path = BASE_DIR / "cbow_best.pt"
    vector_path = BASE_DIR / "word2vec.txt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "word2index": dataset.word2index,
            "index2word": dataset.index2word,
            "embedding_dim": embedding_dim,
            "window_size": window_size,
            "best_loss": best_loss,
        },
        checkpoint_path,
    )
    save_word_vectors(model, dataset, vector_path)

    print(f"训练完成，最优损失：{best_loss:.6f}")
    print(f"模型已保存到：{checkpoint_path}")
    print(f"词向量已保存到：{vector_path}")

    return model, dataset


if __name__ == "__main__":
    train()
