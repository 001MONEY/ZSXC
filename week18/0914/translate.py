"""加载训练好的 Seq2Seq 模型，将中文句子翻译成英文。"""

from pathlib import Path

import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer

from data_process import Vocabulary, zh_tokenizer
from model import Decoder, Encoder, Seq2Seq


MAX_LENGTH = 30
EN_DETOKENIZER = TreebankWordDetokenizer()


def load_translation_model(model_path, zh_vocab, en_vocab, device):
    """根据 checkpoint 中的参数恢复模型。"""
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    encoder = Encoder(
        vocab_size=len(zh_vocab),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        pad_index=Vocabulary.PAD_INDEX,
        dropout=checkpoint.get("dropout", 0.0),
    )
    decoder = Decoder(
        vocab_size=len(en_vocab),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        pad_index=Vocabulary.PAD_INDEX,
        dropout=checkpoint.get("dropout", 0.0),
    )
    model = Seq2Seq(encoder, decoder).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 推理时必须关闭 Dropout。
    model.eval()
    return model


def translate(sentence, model, zh_vocab, en_vocab, device):
    """编码一句中文，自回归生成英文，再将编号解码成单词。"""
    src_ids = zh_vocab.encode(
        sentence,
        tokenizer_fun=zh_tokenizer,
        add_sos=False,
        add_eos=True,
    )

    src = torch.tensor(
        [src_ids],
        dtype=torch.long,
        device=device,
    )
    src_lengths = torch.tensor([len(src_ids)], dtype=torch.long)

    generated_ids = model.generate(
        src,
        src_lengths,
        sos_index=Vocabulary.SOS_INDEX,
        eos_index=Vocabulary.EOS_INDEX,
        max_length=MAX_LENGTH,
    )

    en_tokens = en_vocab.decode(generated_ids[0].cpu().tolist())
    return EN_DETOKENIZER.detokenize(en_tokens)


def main():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "best_seq2seq.pt"
    zh_vocab_path = base_dir / "zh_train_word2index.json"
    en_vocab_path = base_dir / "en_train_word2index.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zh_vocab = Vocabulary.load(zh_vocab_path)
    en_vocab = Vocabulary.load(en_vocab_path)
    model = load_translation_model(
        model_path,
        zh_vocab,
        en_vocab,
        device,
    )

    print("推理设备：", device)
    print("输入中文进行翻译，直接回车退出。")

    while True:
        sentence = input("中文：").strip()
        if not sentence:
            break

        result = translate(
            sentence,
            model,
            zh_vocab,
            en_vocab,
            device,
        )
        print("英文：", result)


if __name__ == "__main__":
    main()
