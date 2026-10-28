"""06：把模型包装成 FastAPI 接口（带重试 + 备用模型）。

课堂版（原名 teste6.py）的三个问题，这里都修掉了：

1. `response.status_code` —— AIMessage 没有这个属性，走到就 AttributeError
2. 没有 try/except —— 网络异常时 model.invoke 直接抛异常，备用模型永远不会被调用
3. 循环 3 次后 response 可能仍是 None，接口会返回 {"ai_content": null}

另外：接口名从 /users 改成语义更准的 /chat，监听 127.0.0.1（本机自用，不对外网开放）。
备用模型从课堂的本地 Ollama 改成 DeepSeek 云端接口（复用 .env 里的 OPENAI_*）。

启动：python 06_chat_server_fastapi.py
调用：python 07_chat_client.py
"""

import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from langchain.chat_models import init_chat_model

from lab_config import API_KEY, BASE_URL, MAX_TOKENS, MODEL, PROVIDER

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

RETRIES = 3

# 主模型
primary = init_chat_model(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    max_tokens=MAX_TOKENS,
    model_provider="openai",
)

# 备用模型：DeepSeek（复用 .env 里的 OPENAI_* 配置），主模型挂掉时兜底
backup = None
backup_model = os.getenv("OPENAI_MODEL", "deepseek-flash")
backup_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
backup_key = os.getenv("OPENAI_API_KEY")

if PROVIDER == "openai":
    # 主模型本来就是 OPENAI_* 指向的服务，再兜底也没意义
    logger.warning("主模型已使用 OPENAI_* 配置，不重复配置 DeepSeek 备用模型")
elif not backup_key:
    logger.warning("未配置 OPENAI_API_KEY，本次没有备用模型")
else:
    backup = init_chat_model(
        model=backup_model,
        base_url=backup_url,
        api_key=backup_key,
        max_tokens=MAX_TOKENS,
        model_provider="openai",
    )

logger.info("主模型：%s @ %s", MODEL, BASE_URL)
if backup is not None:
    logger.info("备用模型：%s @ %s", backup_model, backup_url)


def invoke_with_retry(messages: list) -> str:
    """先重试主模型，仍失败则切备用模型；全挂了抛 502。"""
    last_error: Exception | None = None

    for attempt in range(1, RETRIES + 1):
        try:
            response = primary.invoke(messages)
            if response.content:
                logger.info("主模型成功（第 %d 次尝试）", attempt)
                return response.content
            logger.warning("主模型返回空内容（第 %d 次尝试）", attempt)
        except Exception as exc:  # 网络超时、鉴权失败、限流都走这里
            last_error = exc
            logger.error("主模型调用失败（第 %d 次尝试）：%s", attempt, exc)

    if backup is not None:
        try:
            logger.warning("主模型重试耗尽，切换到备用模型（DeepSeek）")
            response = backup.invoke(messages)
            if response.content:
                logger.info("备用模型成功")
                return response.content
            logger.error("备用模型返回空内容")
        except Exception as exc:
            last_error = exc
            logger.error("备用模型也失败：%s", exc)

    raise HTTPException(status_code=502, detail=f"模型调用失败：{last_error}")


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    messages = data.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="请求体缺少 messages 字段")
    return {"ai_content": invoke_with_retry(messages)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
