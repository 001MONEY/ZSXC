#子代理

from deepagents import create_deep_agent

from langchain_ollama import ChatOllama

from langchain.agents import create_agent

def weather_search(query: str) -> str:
    """搜索天气。"""
    return (
        f"{query}的天气是：晴天"
    )


researcher = {
    "name": "researcher",
    "description": "用来执行天气查询的子代理",
    "system_prompt": "你只做天气查询。",
    "tools": [weather_search],
}

agent = create_deep_agent(
    model=ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434",reasoning=True),
    subagents=[researcher]
)

print(agent.invoke({"messages":[{"role": "user", "content": "北京天气如何？"}]}))


