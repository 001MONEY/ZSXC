from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage

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


human_message = HumanMessage(content="你是谁")  
system_message = SystemMessage(content="你是一名文学家，你擅长模仿别人的风格产出文章。你必须以json格式输出结果。json格式如下：{'title': 文章标题, 'content': 文章内容}")
ai_message = AIMessage(content="我是千问")

new_message = HumanMessage(content="模仿朝花夕拾的风格产出文章")

messages = [system_message, human_message, ai_message,new_message]
response = model.invoke(messages)
print(response)
print(type(response))