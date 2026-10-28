"""课堂代码的本地化配置（week23/20261027 共用）。

目的：脚本里不再出现硬编码的 API Key / 别人的局域网 IP，全部从 week23/.env 读取。

切换模型来源：改 .env 里的 LLM_PROVIDER
    openai —— 任意 OpenAI 兼容云端接口（DeepSeek、百炼、智谱…）
    qwen   —— 阿里云百炼千问（OpenAI 兼容模式）
    ollama —— 本机 Ollama（免费）

用法：
    from lab_config import get_model, show_config

    model = get_model()
    print(model.invoke([{"role": "user", "content": "你是谁"}]).content)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 配置文件在上一级（week23/.env），和 Notebook 用的是同一份
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# override=True：改完 .env 直接生效，不会被 shell 里的旧变量挡住
load_dotenv(ENV_FILE, override=True)

PROVIDER = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()


def _need(name: str, default: str | None = None) -> str:
    value = os.getenv(name) or default
    if not value:
        raise RuntimeError(
            f"缺少环境变量 {name}。请在 {ENV_FILE} 里填写，格式参考 .env.example"
        )
    return value.strip()


# ---- 按 provider 解析出统一的 (API_KEY, BASE_URL, MODEL) ----
if PROVIDER == "qwen":
    API_KEY = _need("QWEN_API_KEY")
    BASE_URL = _need("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    MODEL = _need("QWEN_MODEL", "qwen-plus")
elif PROVIDER == "ollama":
    # Ollama 的 OpenAI 兼容层在 /v1，api_key 随便填
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    API_KEY = "ollama"
    BASE_URL = f"{OLLAMA_URL}/v1"
    MODEL = _need("OLLAMA_MODEL", "qwen3.5:2b")
elif PROVIDER == "openai":
    API_KEY = _need("OPENAI_API_KEY")
    BASE_URL = _need("OPENAI_BASE_URL", "https://api.deepseek.com")
    MODEL = _need("OPENAI_MODEL", "deepseek-chat")
else:
    raise RuntimeError(
        f"未知的 LLM_PROVIDER={PROVIDER!r}，可选值：openai / qwen / ollama"
    )

MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))

# 千问的思考模式：开着会先消耗大量 reasoning token，
# max_tokens 偏小时会出现「思考把额度花光、正文为空」（finish_reason=length）
THINKING = (os.getenv("QWEN_ENABLE_THINKING") or "false").strip().lower() in {
    "1", "true", "yes", "on"
}


def get_model(**overrides):
    """按统一配置创建 ChatModel，参数可覆盖。

    云端走 OpenAI 兼容协议；本地 Ollama 用原生接口，并默认关闭 thinking
    （qwen3.5 系列默认开思考链，实测 2B 模型单次请求从 0.5s 涨到 79s）。
    """
    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=overrides.pop("model", MODEL),
            base_url=OLLAMA_URL,
            reasoning=False,
            num_predict=overrides.pop("max_tokens", MAX_TOKENS),
            **overrides,
        )

    from langchain.chat_models import init_chat_model

    params = {
        "model": MODEL,
        "base_url": BASE_URL,
        "api_key": API_KEY,
        "max_tokens": MAX_TOKENS,
        "model_provider": "openai",
    }
    # 只有千问网关认识 enable_thinking；DeepSeek 等普通接口传了可能报错
    if PROVIDER == "qwen":
        params["extra_body"] = {"enable_thinking": THINKING}
    params.update(overrides)
    return init_chat_model(**params)


def show_config() -> None:
    """打印当前配置（Key 打码），排查问题先看这个。"""
    key = API_KEY
    masked = f"{key[:6]}***{key[-4:]}" if len(key) > 12 else "***"
    print(f"provider = {PROVIDER}")
    print(f"model    = {MODEL}")
    print(f"base_url = {BASE_URL}")
    print(f"api_key  = {masked}")
    print(f"max_tokens = {MAX_TOKENS}")
    if PROVIDER == "qwen":
        print(f"thinking   = {THINKING}（true 会白烧 reasoning token）")


if __name__ == "__main__":
    show_config()
