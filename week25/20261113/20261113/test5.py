#构建好的中间键
from langchain.agents import create_agent

from langgraph.checkpoint.memory import MemorySaver

from langchain_ollama import ChatOllama

from langchain.agents.middleware import SummarizationMiddleware,dynamic_prompt,ModelRequest
model1 = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434",reasoning= False)

checkpoint = MemorySaver()

@dynamic_prompt
def dynamic_prompt_template(request: ModelRequest) -> str:
    messages = request.system_message.content
    print(messages)

    return "你是李四，你是一个好人"
agent = create_agent(
    model=model1,
    tools=[],
    system_prompt="你是张三，你是一个好人",
    middleware=[SummarizationMiddleware(model=model1, #用那个模型做摘呀
                     trigger=("messages",20), #超过5条做摘要
                     keep=("messages",10)),dynamic_prompt_template]
)
while True:
    messages = input("请输入你的问题:")
    result = agent.invoke({"messages":[{"role": "user", "content": messages}]})
    print(result)
