"""
训练入口脚本

训练流程：
    1. 加载数据、构建词表和 DataLoader
    2. 构建 Seq2Seq 模型（双向2层 GRU Encoder + 单向2层 GRU Decoder）并初始化权重
    3. 训练循环：Teacher Forcing 线性衰减（1.0→0.5，不低于0.5）
    4. AdamW 优化器 + 固定学习率（无调度器）
    5. Early Stopping：验证集 loss 连续 PATIENCE 轮不降则停止
    6. 保存最佳 checkpoint
    7. 训练结束后自动加载最佳模型
    8. TensorBoard 实时记录 loss、翻译示例

查看 TensorBoard：
    tensorboard --logdir=runs
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from config import Config, get_device

from data import Vocabulary, zh_tokenizer, generate_dataloader
from model import build_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import set_seed, translate_sentence


# ============================================================
#  训练与验证
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, clip,
                tf_ratio, epoch_num, total_epochs):
    """
    训练一个 epoch。

    Args:
        model:     Seq2Seq 模型
        loader:    训练 DataLoader
        optimizer: 优化器
        criterion: 损失函数
        device:    计算设备
        clip:      梯度裁剪阈值
        tf_ratio:  Teacher Forcing 比率
        epoch_num: 当前 epoch 编号
        total_epochs: 总 epoch 数

    Returns:
        平均训练 loss
    """
    model.train()
    total_loss, num_batches = 0.0, 0

    pbar = tqdm(loader, desc=f"训练 Epoch {epoch_num}/{total_epochs}", leave=False)
    for src, tgt, src_lengths in pbar:
        src, tgt = src.to(device), tgt.to(device)

        # 前向传播
        output = model(src, tgt, src_lengths=src_lengths, teacher_forcing_ratio=tf_ratio)

        # 计算损失：跳过第0位（SOS 位置），只计算实际预测部分
        # tgt: [<SOS>, 10, 19, 20, 31, 45, <EOS>]   -- 真实标签
        # output: 模型预测的标签虽然没有<SOS>， 但是模型在训练的时候设置的output大小和tgt是一样的
        # 所以第一个索引出被<SOS>占位了
        # 下面的reshape是为了将output的shape做成（N, C）， tgt做成(N)的形状
        # output的shape: (batch, tgt_len, vocab_size)
        # output[:, 1:]的shape: (batch, tgt_len-1, vocab_size)
        # output[:, 1:].reshape(-1, output.size(-1)的shape: (batch*tgt_len-1, vocab_size)
        # tgt的shape: (batch, tgt_len)
        # tgt[:, 1:]的shape: (batch, tgt_len-1)   (N, C)
        # tgt[:, 1:].reshape(-1)的shape: (batch*tgt_len-1)  (N)
        loss = criterion(output[:, 1:].reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # 实时显示当前 loss 和平均 loss
        avg_loss = total_loss / max(num_batches, 1)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{avg_loss:.4f}'})

    pbar.close()
    return total_loss / max(num_batches, 1)


def evaluate(model, loader, criterion, device):
    """
    在验证集上评估模型。

    使用 Teacher Forcing（teacher_forcing_ratio=1.0）：给定真实前缀衡量模型的
    条件似然，与训练 loss 口径一致，两条曲线可直接比较，early stopping 更稳定。
    无 Teacher Forcing 的真实生成质量由训练中的翻译示例和 evaluate.py 的 BLEU 评估。

    Args:
        model:     Seq2Seq 模型
        loader:    验证 DataLoader
        criterion: 损失函数
        device:    计算设备

    Returns:
        平均验证 loss
    """
    model.eval()
    total_loss, num_batches = 0.0, 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="验证中", leave=False)
        for src, tgt, src_lengths in pbar:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt, src_lengths=src_lengths, teacher_forcing_ratio=1.0)  # 与训练 loss 同口径
            loss = criterion(output[:, 1:].reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
            num_batches += 1

            # 实时显示验证 loss
            avg_loss = total_loss / max(num_batches, 1)
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}', 'avg': f'{avg_loss:.4f}'})

        pbar.close()

    return total_loss / max(num_batches, 1)


# ============================================================
#  主训练流程
# ============================================================

def train():
    """
    完整训练流程。

    Args:
        cfg: 全局配置

    Returns:
        model: 加载了最佳权重的模型
        src_vocab, tgt_vocab, val_pairs: 训练产出
    """
    # 设置随机种子和设备
    set_seed(Config.SEED)
    device = get_device()
    print(f"计算设备: {device}")

    # ── 数据准备 ──
    train_loader, val_loader, train_pairs, val_pairs, src_vocab, tgt_vocab = generate_dataloader()

    # ── 模型构建 ──
    model = build_model(len(src_vocab.vocab), len(tgt_vocab.vocab), device)

    # ── 损失函数 ──
    # ignore_index=Vocabulary.PAD_INDEX 表示<PAD>不参与计算损失
    criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD_INDEX)

    # ── 优化器 ──
    # AdamW：将权重衰减从梯度更新中解耦，正则化效果优于 Adam
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    # 学习率调度：验证 loss 连续不降时把学习率减半，缓解固定学习率后期的过拟合/震荡
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.LR_DECAY_FACTOR,
        patience=Config.LR_DECAY_PATIENCE,
        min_lr=1e-5,
    )

    # ── TensorBoard ──
    writer = SummaryWriter(log_dir=Config.TENSORBOARD_DIR)

    # ── 训练循环 ──
    best_val_loss = float("inf")
    patience_counter = 0  # 早停计数器
    # 确保权重保存目录存在（全新环境首次训练时 weights/ 可能不存在）
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    best_checkpoint_path = os.path.join(Config.SAVE_DIR, Config.CHECKPOINT_NAME)

    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)

    for epoch in range(Config.NUM_EPOCHS):
        t0 = time.time()

        # Teacher Forcing 线性衰减：从 TF_RATIO_START 线性降到 TF_RATIO_END，但不低于 TF_RATIO_END
        # 前期 TF≈1.0 快速收敛，后期 TF 逐步降低学习纠错能力
        # cfg.TF_RATIO_START: 起始值1.0
        # cfg.TF_RATIO_END: 起始值0.5
        # cfg.TF_RATIO_START - cfg.TF_RATIO_END: 衰减总量
        # epoch / cfg.NUM_EPOCHS： 当前训练轮次进度
        # (cfg.TF_RATIO_START - cfg.TF_RATIO_END) * (epoch / cfg.NUM_EPOCHS) 总的衰减根据进度分配
        tf_ratio = max(
            Config.TF_RATIO_END,
            Config.TF_RATIO_START - (Config.TF_RATIO_START - Config.TF_RATIO_END) * (epoch / Config.NUM_EPOCHS)
        )

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 Config.CLIP, tf_ratio, epoch + 1, Config.NUM_EPOCHS)
        val_loss = evaluate(model, val_loader, criterion, device)

        # 学习率调度：val 连续 LR_DECAY_PATIENCE 轮不降就把学习率减半
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0
        tqdm.write(
            f"Epoch {epoch + 1:02d}/{Config.NUM_EPOCHS} | "
            f"train: {train_loss:.4f} | val: {val_loss:.4f} | "
            f"tf: {tf_ratio:.2f} | lr: {current_lr:.2e} | {elapsed:.1f}s"
        )

        # ── TensorBoard 记录 ──
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("TeacherForcing/ratio", tf_ratio, epoch)
            writer.add_scalar("LR", current_lr, epoch)

        # 每5轮展示翻译示例，并写入 TensorBoard
        if (epoch + 1) % 5 == 0:
            tqdm.write("  翻译示例：")
            examples_lines = []
            for zh, en in val_pairs[:3]:
                pred = translate_sentence(model, zh, src_vocab, tgt_vocab, zh_tokenizer, device)
                line = f"{zh} → {pred}"
                tqdm.write(f"    {line}")
                examples_lines.append(f"<b>中文:</b> {zh}<br><b>参考:</b> {en}<br><b>翻译:</b> {pred}")
            if writer:
                writer.add_text(
                    f"Examples/epoch_{epoch + 1:02d}",
                    "<hr>".join(examples_lines),
                    epoch,
                )

        # Early Stopping + Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                tqdm.write(f"\nEarly stopping at epoch {epoch + 1}")
                break

    if writer:
        writer.close()

    # ── 加载最佳模型 ──
    # ── 如果训练完成后就立即评估模型， 那么可以在训练完成后加载下面的代码让模型加载best.pt进行评估 ──
    try:
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    tqdm.write(f"\n已加载最佳 weights (val_loss: {best_val_loss:.4f})")

    return model, src_vocab, tgt_vocab, val_pairs


# ============================================================
#  运行入口
# ============================================================

if __name__ == "__main__":
    train()
