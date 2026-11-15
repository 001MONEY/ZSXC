from __future__ import annotations


from pathlib import Path

from deepagents import create_deep_agent

from deepagents.backends import CompositeBackend,StateBackend,StoreBackend

from langchain_ollama import ChatOllama

from langgraph.store.memory import InMemoryStore

from langgraph.checkpoint.memory import MemorySaver

Root = Path(__file__).resolve().parent

store = InMemoryStore()
checkpoint = MemorySaver()


store.put(
    ("classroom",),
    "/AGENTS.md",
    {
        "content": "## 回答风格\n- 默认用简短的几句话\n",
        "encoding": "utf-8",  #必须要指明编码
    },
)


agent  = create_deep_agent(

    model  =ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434",reasoning=True),
    store=store,
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(namespace=lambda rt: ("classroom",))
        }),
    memory=["/memories/AGENTS.md"]
    )



