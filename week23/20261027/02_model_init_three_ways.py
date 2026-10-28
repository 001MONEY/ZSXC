"""02：三种创建模型的方式对比（本地化版）。

1) ChatOpenAI      —— OpenAI 兼容接口（云端：DeepSeek / 百炼 / 智谱…）
2) ChatOllama      —— Ollama 原生接口（本地，免费）
3) init_chat_model —— 统一工厂，按 model_provider 创建

课堂版把 key 和老师机器的局域网 IP（192.168.1.105）写死在注释里，
这里全部改成从 week23/.env 读取，换机器/换模型都不用改代码。
"""
import os

from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from lab_config import API_KEY, BASE_URL, MAX_TOKENS, MODEL, PROVIDER

messages = [{"role": "user", "content": "你是谁"}]


def demo(title: str, model) -> None:
    print("=" * 15, title, "=" * 15)
    response = model.invoke(messages)
    print(type(response))
    print(response.content)
    print()

# 1) 云端：直接用 ChatOpenAI（底层就是 OpenAI 协议）
demo(
    "ChatOpenAI",
    ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, max_tokens=MAX_TOKENS),
)
# 2) 本地：ChatOllama 走 Ollama 原生 /api/chat，不是 OpenAI 兼容层
if PROVIDER == "ollama":
    demo(
        "ChatOllama",
        ChatOllama(
            model=MODEL,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            num_predict=MAX_TOKENS,  # 注意：Ollama 原生接口用 num_predict，不是 max_tokens
        ),
    )
else:
    print("（跳过 ChatOllama：把 .env 的 LLM_PROVIDER 改成 ollama 才会跑）\n")

# 3) init_chat_model：一个工厂函数，换 provider 只改参数
demo(
    "init_chat_model",
    init_chat_model(
        model=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        max_tokens=MAX_TOKENS,
        model_provider="openai",
    ),
)
