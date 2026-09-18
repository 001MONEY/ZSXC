"""训练基础 RNN Seq2Seq 中英翻译模型。"""

from pathlib import Path

import torch
from torch import nn

from config import Config
from data_process import Vocabulary, load_cmn_to_pairs
from dataset import create_train_val_dataloaders, load_or_create_split
from model import Decoder, Encoder, Seq2Seq


# 训练参数
BATCH_SIZE = 32
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 1
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
DROPOUT = 0.2
LR_FACTOR = 0.5
LR_PATIENCE = 2
MIN_LEARNING_RATE = 1e-5
CLIP_NORM = 1.0


def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    """训练一轮，返回所有有效 Token 的平均损失。"""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        src = batch["src"].to(device)
        src_lengths = batch["src_lengths"]
        decoder_input = batch["decoder_input"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        logits = model(src, src_lengths, decoder_input)
        loss = loss_function(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

        valid_tokens = target.ne(Vocabulary.PAD_INDEX).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model, dataloader, loss_function, device):
    """在验证集上计算平均损失，不更新模型参数。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        src = batch["src"].to(device)
        src_lengths = batch["src_lengths"]
        decoder_input = batch["decoder_input"].to(device)
        target = batch["target"].to(device)

        logits = model(src, src_lengths, decoder_input)
        loss = loss_function(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
        )

        valid_tokens = target.ne(Vocabulary.PAD_INDEX).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    return total_loss / total_tokens


def save_model(model, file_path):
    """保存模型参数和创建模型所需的基本配置。"""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": EMBEDDING_DIM,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        },
        file_path,
    )


def main():
    base_dir = Path(__file__).resolve().parent
    split_path = base_dir / "train_val_split.json"
    zh_vocab_path = base_dir / "zh_train_word2index.json"
    en_vocab_path = base_dir / "en_train_word2index.json"
    model_path = base_dir / "best_seq2seq.pt"

    # 1. 加载固定的数据划分和训练专用词表。
    pairs = load_cmn_to_pairs(Config.FILE_PATH)
    train_pairs, val_pairs = load_or_create_split(pairs, split_path)
    zh_vocab = Vocabulary.load(zh_vocab_path)
    en_vocab = Vocabulary.load(en_vocab_path)

    _, _, train_loader, val_loader = create_train_val_dataloaders(
        train_pairs,
        val_pairs,
        zh_vocab,
        en_vocab,
        batch_size=BATCH_SIZE,
    )

    # 2. 创建 Encoder、Decoder 和完整 Seq2Seq。
    encoder = Encoder(
        vocab_size=len(zh_vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        pad_index=Vocabulary.PAD_INDEX,
        dropout=DROPOUT,
    )
    decoder = Decoder(
        vocab_size=len(en_vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        pad_index=Vocabulary.PAD_INDEX,
        dropout=DROPOUT,
    )
    model = Seq2Seq(encoder, decoder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # <PAD> 位置不参与损失计算。
    loss_function = nn.CrossEntropyLoss(
        ignore_index=Vocabulary.PAD_INDEX
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    # 验证损失长期不再下降时，自动减小学习率。
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )

    # 3. 训练并保存验证损失最小的模型。
    best_val_loss = float("inf")
    print("训练设备：", device)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )
        val_loss = evaluate(
            model,
            val_loader,
            loss_function,
            device,
        )
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"lr={current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, model_path)
            print("保存最佳模型：", model_path)


if __name__ == "__main__":
    main()
