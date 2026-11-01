from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.runnables.history import RunnableWithMessageHistory

sys.path.insert(0, str(Path(__file__).resolve().parent / "20261027"))
from lab_config import ENV_FILE, MODEL, PROVIDER, get_model  # type: ignore  # noqa: E402

load_dotenv(ENV_FILE, override=True)

# ---- 长度限制与压缩参数（都能用 .env 覆盖）----
MAX_TURNS = int(os.getenv("CHAT_MAX_TURNS", "20"))          # 最大轮数
MAX_MESSAGES = MAX_TURNS * 2                                # 40 条消息
KEEP_RECENT = int(os.getenv("CHAT_KEEP_RECENT", "20"))      # 压缩后保留的最近消息条数
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "300"))
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "400"))
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
SESSION_TTL = int(os.getenv("CHAT_SESSION_TTL", str(7 * 24 * 3600)))  # 7 天

ROLE_CN = {"human": "用户", "ai": "助手", "system": "系统", "tool": "工具"}


# ============================================================
# 1. 存储层：Redis / 内存，接口一致
# ============================================================
def empty_doc() -> dict:
    return {"summary": "", "messages": []}


class HistoryStore:
    """会话历史的持久化接口。doc 结构见模块开头的说明。"""

    name = "base"

    def load(self, session_id: str) -> dict:
        raise NotImplementedError

    def save(self, session_id: str, doc: dict) -> None:
        raise NotImplementedError

    def delete(self, session_id: str) -> None:
        raise NotImplementedError


class RedisStore(HistoryStore):
    """优先方案：历史存 Redis，重启进程也不丢。"""

    name = "redis"

    def __init__(self, url: str = REDIS_URL, prefix: str = "chat:history:", ttl: int = SESSION_TTL):
        import redis  # 延迟导入：没装 redis 包也不影响内存模式

        self.client = redis.from_url(url, decode_responses=True)
        self.client.ping()          # 连不上就在这里抛异常，由 create_store 决定是否降级
        self.prefix = prefix
        self.ttl = ttl
        self.url = url

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    def load(self, session_id: str) -> dict:
        raw = self.client.get(self._key(session_id))
        return empty_doc() if not raw else json.loads(raw)

    def save(self, session_id: str, doc: dict) -> None:
        self.client.set(self._key(session_id), json.dumps(doc, ensure_ascii=False), ex=self.ttl)

    def delete(self, session_id: str) -> None:
        self.client.delete(self._key(session_id))


class MemoryStore(HistoryStore):
    """降级方案：存进程内存，接口与 Redis 完全一致（同样存 JSON 字符串）。"""

    name = "memory"

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def load(self, session_id: str) -> dict:
        raw = self._data.get(session_id)
        return empty_doc() if not raw else json.loads(raw)

    def save(self, session_id: str, doc: dict) -> None:
        self._data[session_id] = json.dumps(doc, ensure_ascii=False)

    def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)


def create_store(url: str | None = None, quiet: bool = False) -> HistoryStore:
    """优先 Redis，连不上就降级为内存，并说清代价。"""
    try:
        store = RedisStore(url or REDIS_URL)
        if not quiet:
            print(f"[存储] Redis 已连接：{store.url}")
        return store
    except Exception as exc:  # noqa: BLE001  —— 连接失败 / 没装 redis 包，都要降级
        if not quiet:
            print(f"[存储] Redis 不可用（{type(exc).__name__}: {exc}）")
            print("       → 降级为内存存储，重启进程后历史会丢")
            print("       → 想用 Redis：docker run -d --name redis-chat -p 6379:6379 redis:7-alpine")
        return MemoryStore()


# ============================================================
# 2. 摘要层：小参数模型
# ============================================================
SUMMARY_PROMPT = """你在为一段多轮对话做摘要压缩，摘要会替代原始对话继续参与后续推理。

请把下面内容压缩成不超过 {max_chars} 字的中文摘要：
1. 保留用户的事实信息（姓名、身份、偏好、明确说过的约束）
2. 保留已确认的结论，以及尚未完成的任务和用户最新的诉求
3. 省略寒暄、重复确认、客套话
4. 用第三人称陈述，直接给结论，不要评论

【已有摘要】
{previous_summary}

【新增对话】
{conversation}

只输出摘要正文，不要加标题或解释。"""


def get_summarizer():
    """摘要用的小模型。

    优先读 .env 的 SUMMARY_MODEL（例如更便宜的 flash / 小参数模型）；
    没配就复用主模型，但把 max_tokens 压小，避免摘要本身花太多钱。
    """
    summary_model = os.getenv("SUMMARY_MODEL") or MODEL
    return get_model(model=summary_model, max_tokens=SUMMARY_MAX_TOKENS, temperature=0.3)


# ============================================================
# 3. 记忆层：20 轮上限 + 摘要压缩
# ============================================================
class SummarizingHistory(BaseChatMessageHistory):
    """带长度限制与摘要压缩的会话历史。

    - 存储里只放 `summary` + 最近若干条原始消息
    - 追加后条数 > max_messages（默认 40）时，把前面的消息压成摘要
    - 读出来的 `messages` 会把摘要包成 SystemMessage 放在最前面
    """

    def __init__(
        self,
        session_id: str,
        store: HistoryStore,
        summarizer=None,
        *,
        max_messages: int = MAX_MESSAGES,
        keep_recent: int = KEEP_RECENT,
        max_summary_chars: int = SUMMARY_MAX_CHARS,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self.store = store
        self.summarizer = summarizer
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.max_summary_chars = max_summary_chars
        self.verbose = verbose

    # ---------- LangChain 要求实现的三个成员 ----------
    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        doc = self.store.load(self.session_id)
        history = messages_from_dict(doc["messages"])
        if doc["summary"]:
            history = [SystemMessage(content=f"【早前对话摘要】{doc['summary']}")] + history
        return history

    def add_messages(self, messages) -> None:
        doc = self.store.load(self.session_id)
        doc["messages"].extend(message_to_dict(m) for m in messages)

        if len(doc["messages"]) > self.max_messages:
            doc = self._compress(doc)

        self.store.save(self.session_id, doc)

    def clear(self) -> None:
        self.store.delete(self.session_id)

    # ---------- 统计信息（演示/调试用）----------
    def stats(self) -> dict:
        doc = self.store.load(self.session_id)
        return {
            "session_id": self.session_id,
            "stored_messages": len(doc["messages"]),
            "has_summary": bool(doc["summary"]),
            "summary_chars": len(doc["summary"]),
            "prompt_messages": len(self.messages),
        }

    # ---------- 内部：压缩 ----------
    def _compress(self, doc: dict) -> dict:
        raw = doc["messages"]
        old, recent = raw[: -self.keep_recent], raw[-self.keep_recent:]

        summary = self._summarize(doc["summary"], self._render(old))

        if self.verbose:
            print(
                f"\n[摘要压缩] session={self.session_id} "
                f"{len(raw)} 条 → 摘要({len(summary)} 字) + 最近 {len(recent)} 条"
            )

        return {"summary": summary, "messages": recent}

    def _summarize(self, previous_summary: str, conversation: str) -> str:
        if self.summarizer is None:
            return self._fallback_summary(previous_summary, conversation)

        prompt = SUMMARY_PROMPT.format(
            max_chars=self.max_summary_chars,
            previous_summary=previous_summary or "（无）",
            conversation=conversation,
        )
        try:
            text = self.summarizer.invoke(prompt).content.strip()
            if text:
                return text
            print("[warn] 摘要模型返回空内容，退化为拼接式压缩")
        except Exception as exc:  # noqa: BLE001  —— 网络/额度问题不能拖垮主流程
            print(f"[warn] 摘要模型调用失败（{exc}），退化为拼接式压缩")
        return self._fallback_summary(previous_summary, conversation)

    def _fallback_summary(self, previous_summary: str, conversation: str) -> str:
        """兜底：截取早期对话的尾部若干字，至少不让历史无限增长。"""
        merged = f"{previous_summary}\n{conversation}".strip()
        return merged[-self.max_summary_chars:]

    @staticmethod
    def _render(msgs: list[dict]) -> str:
        lines = []
        for item in msgs:
            msg = messages_from_dict([item])[0]
            role = ROLE_CN.get(msg.type, msg.type)
            lines.append(f"{role}：{msg.content}")
        return "\n".join(lines)


# ============================================================
# 4. 业务层：链 + CLI
# ============================================================
CHAT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个中文助手，回答简洁准确。"
        "如果对话开头带有【早前对话摘要】，把它当作已知背景，不要重复询问已经说过的事。",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])


def build_chatbot(store: HistoryStore, summarizer=None, *, verbose: bool = True):
    """返回一个带记忆的链：invoke 时传 session_id，自动读写历史。"""
    chain = CHAT_PROMPT | get_model(max_tokens=512)

    def get_history(session_id: str) -> SummarizingHistory:
        return SummarizingHistory(session_id, store, summarizer, verbose=verbose)

    return RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="input",       # 用户输入放在哪个键
        history_messages_key="history",   # 历史填进哪个占位符
    )


def ask(chatbot, question: str, session_id: str = "homework-demo") -> str:
    """问一句，返回回答文本。"""
    response = chatbot.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )
    return response.content


def main() -> None:
    store = create_store()
    summarizer = get_summarizer()

    print(
        f"[配置] provider={PROVIDER} model={MODEL} "
        f"上限={MAX_TURNS} 轮/{MAX_MESSAGES} 条 保留最近={KEEP_RECENT} 条"
    )
    print("输入内容开始聊天，空行或 exit 退出\n")

    session_id = os.getenv("CHAT_SESSION_ID", "homework-demo")
    chatbot = build_chatbot(store, summarizer)

    while True:
        question = input("你：").strip()
        if not question or question.lower() in {"exit", "quit"}:
            break
        print("AI：", ask(chatbot, question, session_id))
        stats = SummarizingHistory(session_id, store, verbose=False).stats()
        print(f"     （历史：{stats['stored_messages']} 条，摘要：{'有' if stats['has_summary'] else '无'}）\n")


# ============================================================
# 5. 离线自测：不调用任何模型，验证长度限制与压缩
# ============================================================
class CountingSummarizer:
    """假摘要器：记录被调用次数，返回固定长度摘要。"""

    def __init__(self) -> None:
        self.calls = 0
        self.last_prompt = ""

    def invoke(self, prompt: str):
        self.calls += 1
        self.last_prompt = prompt

        class _Resp:
            content = f"【假摘要 #{self.calls}】用户此前讨论过若干问题，助手已给出结论。" * 3

        return _Resp()


def demo(turns: int = 25) -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    print("=" * 60)
    print(f"离线自测：模拟 {turns} 轮对话（上限 {MAX_MESSAGES} 条 / 压缩后保留 {KEEP_RECENT} 条）")
    print("=" * 60)

    store = MemoryStore()
    summarizer = CountingSummarizer()
    history = SummarizingHistory("demo-session", store, summarizer)

    for i in range(1, turns + 1):
        history.add_messages([
            HumanMessage(content=f"第 {i} 轮：我的名字是张三，请记住。"),
            AIMessage(content=f"第 {i} 轮：好的，我记住了。"),
        ])
        stats = history.stats()
        print(
            f"第 {i:2d} 轮 → 存储 {stats['stored_messages']:2d} 条，"
            f"摘要 {'有' if stats['has_summary'] else '无'}，"
            f"送模型 {stats['prompt_messages']:2d} 条"
        )

    print("-" * 60)
    final = history.stats()
    print(f"最终存储：{final['stored_messages']} 条（上限 {MAX_MESSAGES}）")
    print(f"摘要长度：{final['summary_chars']} 字，摘要模型被调用 {summarizer.calls} 次")
    print(f"第一次压缩时送给摘要模型的提示词前 200 字：\n{summarizer.last_prompt[:200]}…")
    assert final["stored_messages"] <= MAX_MESSAGES, "超出长度限制！"
    assert summarizer.calls >= 1, "从未触发摘要压缩！"
    print("\n[OK] 断言通过：长度受限、摘要已生成，且早期信息没有丢（摘要里保留了原对话内容）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多轮对话 + 长度限制 + 摘要压缩")
    parser.add_argument("--demo", action="store_true", help="离线自测，不调用模型")
    args = parser.parse_args()

    if args.demo:
        demo()
    else:
        main()
