from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser,PydanticOutputParser
from pydantic import BaseModel, Field

model = init_chat_model(
    model="qwen3.8-max",
    base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
    api_key="sk-ws-H.PXXRYMP.Y6wZ.MEQCIDKz3lBk18q7GfsxvrJoLvFp6TgNTTCPdcuXKsqZisUKAiA1nv-nq4i8djg8xXanm2GsBQ8SJ70dBUIZ4XCsqxl4LA",
    max_tokens=1000,
    temperature=1.2,
    top_p=0.9,
    model_provider="openai",
    extra_body={"enable_thinking": False})



# prompt = PromptTemplate(
#     input_variables=["text"],
#     template='你是一个数据提取助手，请从以下{text}中提取出我们要的内容并且以json格式返回。样例问题：用户：查订单12345678。\n客服：无线蓝牙耳机，199元，已发货，输出格式为：{{"order_id": "12345678", "product_name": "无线蓝牙耳机", "amount": 199.0, "status": "已发货"}}')


# chain = prompt | model | JsonOutputParser()

# res = chain.invoke({"text": "用户：收纳箱单号44002211，仓库拣货了吗？\n客服：收纳箱45.80元，目前待发货，正在拣货。"})
# print(type(res))
# print(res.get("order_id"))



#自定义输出解释器

class OrderInfo(BaseModel):
    order_id: str = Field(description="订单号")
    product_name: str = Field(description="商品名称")
    amount: float = Field(description="金额")
    status: str = Field(description="状态")


output_parser = PydanticOutputParser(pydantic_object=OrderInfo)
print(output_parser.get_format_instructions())


prompt = PromptTemplate(
    input_variables=["text"],
    template='你是一个数据提取助手，请从以下{text}中提取出我们要的内容并且以json格式返回。{format_instructions}')


chain = prompt | model | output_parser

res = chain.invoke({"text": "用户：收纳箱单号44002211，仓库拣货了吗？\n客服：收纳箱45.80元，目前待发货，正在拣货。","format_instructions": output_parser.get_format_instructions()})
print(res)
print(type(res))
print(res.order_id)
print(res.product_name)
print(res.amount)
print(res.status)

print(type(res.model_dump()))