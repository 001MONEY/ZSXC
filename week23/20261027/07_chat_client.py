import requests


url = "http://127.0.0.1:8000/chat"  # 对应 06_chat_server_fastapi.py 的接口

data = {
    "messages": [
        {"role": "system", "content": "你是一名文学家，你擅长诗词歌赋，你的回答必须温文尔雅，用词生动"}
    ]
}

# response = requests.post(url, json=data)
# print(response.json())


def user_chat(messages):
    while True:
        input_message = input("请输入你的问题（回车退出）：").strip()
        if not input_message:
            break
        messages.append({"role": "user", "content": input_message})

        # 修 bug：原来写的是 json=data（全局变量），形参 messages 成了摆设
        response = requests.post(url, json={"messages": messages}, timeout=120)
        response.raise_for_status()
        answer = response.json()["ai_content"]

        print(answer)
        messages.append({"role": "assistant", "content": answer})
        print("-" * 100)


if __name__ == "__main__":
    user_chat(data["messages"])