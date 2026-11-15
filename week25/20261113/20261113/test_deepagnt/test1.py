from __future__ import annotations


from pathlib import Path

from deepagents import create_deep_agent,FilesystemPermission

from deepagents.backends import FilesystemBackend

from langchain_ollama import ChatOllama

Root = Path(__file__).resolve().parent


#这是一段会议聊天
TRANSCRIPT = """
主题没写在标题上。下午临时对了一下上线。
李雷说登录页周五前必须改完文案，王芳负责。
韩梅梅问要不要同时改帮助中心，大家没定。
赵强说他下周三给出回滚方案。
"""

def build_agent():
    model = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434",reasoning=True)
    agent = create_deep_agent(
        model=model,
        backend=FilesystemBackend(root_dir=str(Root),virtual_mode=True),
        skills=["/skills/"],
        permissions=[
            FilesystemPermission(
                operations=["read"],
                paths=["/skills/**"],
                mode="allow",
            )
        ],
        system_prompt="你是会议助手。你是会议助手。"
    "必须先用 read_file 读取 /skills/meeting-notes/SKILL.md。"
    "然后必须调用 write_file，把纪要写到 /notes/meeting.md。"
    "write_file 成功之前，不要把纪要正文当作最终回复。",

    )
    return agent



def main():
    agent = build_agent()
    result = agent.invoke({"messages":[{"role": "user", "content": TRANSCRIPT}]})
    print(result)



if __name__ == "__main__":
    main()