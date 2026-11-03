
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent,AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


model = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434")


prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，可以使用工具帮助用户。"),
    MessagesPlaceholder(variable_name="chat_history"),   # 历史消息
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),  #存放工具结果的
])


class Weather(BaseModel):
    city: str = Field(description="城市名")
    # weather: str = Field(description="天气")
    # temperature: str = Field(description="温度")



@tool(args_schema=Weather)
def get_weather(city: str,) -> Weather:
    """查询指定城市的当前天气。"""

    return f"查询{city}的天气为晴，温度为25.0"


tools = [get_weather]


agent = create_tool_calling_agent(model,tools , prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    handle_parsing_errors=True,#工具调用报错时不抛异常，而是返回一个错误提示让 agent 继续处理
)

res = agent_with_history.invoke({"input": "北京天气怎么样？"},config={"configurable": {"session_id": "user_001"}})
print(res)