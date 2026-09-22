"""临时推理演示：加载 best.pt，批量翻译示例句子（跑完即删）。"""
import torch

from config import Config, get_device
from data import EN_VOCAB_PATH, ZH_VOCAB_PATH, Vocabulary, zh_tokenizer
from model import build_model
from utils import translate_sentence

device = get_device()
src_vocab = Vocabulary.load(ZH_VOCAB_PATH)
tgt_vocab = Vocabulary.load(EN_VOCAB_PATH)

model = build_model(len(src_vocab), len(tgt_vocab), device)
ckpt = Config.CHECKPOINT_PATH
model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
model.eval()
print("已加载:", ckpt, "| device:", device)

sentences = [
    "我爱你。",
    "你叫什么名字？",
    "今天天气很好。",
    "我在学英语。",
    "这个多少钱？",
    "请帮我找一下我的钱包。",
    "你看上去不太好。",
    "汤姆不想让玛丽失望。",
    "富士山顶盖满了雪。",
    "明天我要去北京。",
]

for zh in sentences:
    pred = translate_sentence(model, zh, src_vocab, tgt_vocab, zh_tokenizer, device)
    print(f"{zh}\n    -> {pred}\n")
