from langchain.chat_models import init_chat_model

from lab_config import API_KEY, BASE_URL, MAX_TOKENS, MODEL

# 本地化改造：凭据与模型名来自 week23/.env
model = init_chat_model(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    max_tokens=MAX_TOKENS,
    temperature=1.2,
    top_p=0.9,
    model_provider="openai"
)


messages = [
    {"role": "system", "content": "你是一名文学家，你擅长诗词歌赋，你的回答必须温文尔雅，用词生动"},
]


#多轮对话


def multi_turn_conversation(messages):
    while True:
        input_message = input("请输入你的问题（回车退出）：").strip()
        if not input_message:  # 空回车就退出，避免死循环里一直烧 token
            break
        messages.append({"role": "user", "content": input_message})
        response = model.invoke(messages)
        print(response.content)
        messages.append(response)
        print("-"*100)


if __name__ == "__main__":
    multi_turn_conversation(messages)