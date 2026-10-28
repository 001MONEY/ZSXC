#大模型的访问 肯定是得走 网络 访问
#需要客户端
# openai
#访问线上的大模型 需要 ip地址 还需要apikey
# 本地化改造：Key / base_url / 模型名全部来自 week23/.env，不再硬编码
from openai import OpenAI
from lab_config import API_KEY, BASE_URL, MODEL
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

messages = [{"role": "user", "content": "你是谁"}]
completion = client.chat.completions.create(
    model=MODEL,  # 模型名来自 .env（QWEN_MODEL / OPENAI_MODEL）
    messages=messages,
    extra_body={"enable_thinking": True},
    stream=True
)

# print(completion.choices[0].message.content)

is_answering = True  # 是否进入回复阶段
print("\n" + "=" * 20 + "思考过程" + "=" * 20)
for chunk in completion:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "完整回复" + "=" * 20)
            is_answering = True
        print(delta.content, end="", flush=True)