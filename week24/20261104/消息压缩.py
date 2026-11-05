from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI


def get_weather(city: str) -> str:
    """查询指定城市的当前天气。"""
    return f"{city}：晴，25°C"


def add(a: float, b: float) -> str:
    """计算两个数的和。"""
    return str(a + b)


# 发给模型的说明书。模型只看这份 JSON，不会拿到上面的函数。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "计算两个数的和。",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个数"},
                    "b": {"type": "number", "description": "第二个数"},
                },
                "required": ["a", "b"],
            },
        },
    },
]

FUNCTIONS = {
    "get_weather": get_weather,
    "add": add,
}


class ChatSession:
    """一段会话：记住多轮历史，过长时压缩旧消息，并在本地执行模型要求的工具。"""

    def __init__(self, llm, system="你是一个聊天助手。", trigger=10, keep=4):
        self.llm = llm
        self.system = system
        self.trigger = trigger
        self.keep = keep

        self.summary = ""
        self.history = []

    def ask(self, text: str) -> str:
        self.history.append(HumanMessage(text))
        self._compress()

        reply = self.llm.invoke(self._messages(), tools=TOOLS)
        self.history.append(reply)

        # 模型可能连续要工具。每轮都先在本地执行完，再把结果送回去。
        for _ in range(5):
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                result = self._call_tool(call)
                self.history.append(
                    ToolMessage(content=result, tool_call_id=call["id"])
                )
            reply = self.llm.invoke(self._messages(), tools=TOOLS)
            self.history.append(reply)

        return reply.content

    def _messages(self):
        messages = [SystemMessage(self.system)]
        if self.summary:
            messages.append(SystemMessage(f"更早对话的摘要：\n{self.summary}"))
        messages += self.history
        return messages

    def _compress(self):
        if len(self.history) <= self.trigger:
            return

        cut = len(self.history) - self.keep
        # 从一条用户消息处切开，避免把工具调用和工具结果拆开。
        while cut < len(self.history) and self.history[cut].type != "human":
            cut += 1
        if cut >= len(self.history):
            return

        old, self.history = self.history[:cut], self.history[cut:]
        old_text = "\n".join(self._line(item) for item in old)
        result = self.llm.invoke(
            [
                SystemMessage(
                    "把对话压成简短摘要，保留用户目标、已确定的事实、工具查到的结果和未完成的事。只输出摘要。"
                ),
                HumanMessage(f"已有摘要：\n{self.summary or '无'}\n\n需要并入的对话：\n{old_text}"),
            ]
        )
        self.summary = result.content
        print("\n[已压缩旧对话]\n")

    @staticmethod
    def _line(message):
        if message.type == "human":
            return f"用户: {message.content}"
        if message.type == "tool":
            return f"工具结果: {message.content}"
        calls = ""
        if message.tool_calls:
            calls = " [调用工具: " + ", ".join(call["name"] for call in message.tool_calls) + "]"
        return f"助手: {message.content}{calls}"

    @staticmethod
    def _call_tool(call):
        function = FUNCTIONS.get(call["name"])
        if function is None:
            return f"没有名为 {call['name']} 的工具"
        try:
            return str(function(**call["args"]))
        except Exception as exc:
            return f"工具执行失败: {exc}"


def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    session = ChatSession(llm)
    print("输入 exit 结束。")
    while True:
        text = input("你: ").strip()
        if text in {"exit", "quit"}:
            break
        if not text:
            continue
        print("助手:", session.ask(text))


if __name__ == "__main__":
    main()
