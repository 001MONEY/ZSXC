"""多模态示例：把图片交给视觉模型（kimi-k3.1-pro），让模型转录或描述。

和 01~09 那些文本脚本的区别只有一个：messages 的 content 从「字符串」变成「列表」，
列表里既能放 text，也能放 image_url。

用法：
    python 10_vision_model.py                       # 用内置的图片 URL
    python 10_vision_model.py .\\demo.jpg            # 用本地图片
    python 10_vision_model.py .\\demo.jpg "描述这张图里有什么"  # 自定义提问

配置全部来自 week23/.env：
    DASHSCOPE_API_KEY   （可以写成 ${QWEN_API_KEY}，dotenv 会展开）
    DASHSCOPE_BASE_URL
    VISION_MODEL
"""

from __future__ import annotations

import base64
import mimetypes
import sys
from pathlib import Path

from lab_config import VISION_MODEL, get_vision_client

DEFAULT_IMAGE = (
    "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/ctdzex/biaozhun.jpg"
)
DEFAULT_PROMPT = "请仅输出图像中的文本内容。"


def to_image_url(target: str) -> str:
    """本地文件转 data URL，http(s) 地址原样返回。"""
    if target.startswith(("http://", "https://", "data:")):
        return target
    path = Path(target)
    if not path.is_file():
        raise FileNotFoundError(f"找不到图片：{path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def ask_image(image: str = DEFAULT_IMAGE, prompt: str = DEFAULT_PROMPT, model: str | None = None) -> str:
    client = get_vision_client()
    completion = client.chat.completions.create(
        model=model or VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": to_image_url(image)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return (completion.choices[0].message.content or "").strip()


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    prompt = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PROMPT
    print(f"model = {VISION_MODEL}")
    print(f"image = {image}")
    print("--- 模型输出 ---")
    print(ask_image(image, prompt))
