import requests

base = "http://127.0.0.1:8001"  # 对应 08_fastapi_request_basics.py（8000 是 06 的 /chat）

# params 是 URL 问号后面的查询参数（query）。 常用与get
# requests 会把它拼到地址后面，服务端用 request.query_params 读取。
# 第 8 行发出去的实际地址是：
# http://127.0.0.1:8000/add_one?operatorId=PUoiinWIpxxx

'''参数	放在哪	服务端怎么取
params
URL 的 ? 后面
request.query_params

headers
请求头
request.headers

json
请求体
await request.json()
'''
post_resp = requests.post(
    url=f"{base}/add_one",
    params={"operatorId": "PUoiinWIpxxx"},
    headers={
        "Content-Type": "application/json",
        "x-acs-dingtalk-access-token": "xxxxx",
    },
    json={"number": 41},
    timeout=5,
)
print(post_resp.status_code, post_resp.json())
# {"result": 42}

# GET /hello：query 里带 name、age，头里带 x-token
get_resp = requests.get(
    url=f"{base}/hello",
    params={"name": "小明", "age": 18},
    headers={"x-token": "demo-token"},
    timeout=5,
)
print(get_resp.status_code, get_resp.json())
# {"msg": "你好，小明", "age": "18", "token": "demo-token", "path": "/hello", "method": "GET"}