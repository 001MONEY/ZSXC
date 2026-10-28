"""08：FastAPI 请求解析基础（body / headers / query）+ 容错与日志。

课堂版三条主线（手动解析 / 请求头 / Query）保留，另外补齐工程上必须要有的东西：

1. **请求日志中间件**：每条请求记录 request-id / 方法 / 路径 / 状态码 / 耗时
2. **统一异常出口**：非法 JSON、缺参数、未捕获异常都返回结构化 JSON，堆栈只进日志不外泄
3. **HTTPException**：课堂版写的 `return {"error": ...}, 400` 在 FastAPI 里是无效的
   —— 它会被序列化成 `[{"error": ...}, 400]` 并以 **200** 返回，状态码根本没生效
4. **Pydantic 模型**：另给一个 `/add_one_v2`，非法请求体由框架自动挡成 422

---

原始 HTTP 报文长什么样（课堂上讲的钉钉待办接口）：

    POST /v1.0/todo/users/PUoixxxxGiP6g/tasks?operatorId=PUoiinWIpxxx
    Host: api.dingtalk.com
    x-acs-dingtalk-access-token: xxxxx        # 请求头
    Content-Type: application/json            # 请求头
                                              # 下面才是数据主体
    {"sourceId": "isv_dingtalkTodo1", "subject": "接入钉钉待办", ...}

    路由  = /v1.0/todo/users/{id}/tasks
    query = operatorId=PUoiinWIpxxx
    头    = x-acs-dingtalk-access-token
    body  = JSON

启动：python 08_fastapi_request_basics.py   （监听 127.0.0.1:8001）
调用：python 09_requests_params.py
"""

import logging
import time
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

# ---------- 日志：时间 + 级别 + 模块 + 消息 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("api08")

app = FastAPI(title="08 · FastAPI 请求解析示例")


# ---------- 1. 请求日志中间件：每个请求打一行，带 request-id 和耗时 ----------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    request.state.request_id = request_id          # 后续处理函数读到的是同一个 id
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        # 最外层兜底：保证无论如何都有响应，且堆栈只进日志
        cost_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "[%s] %s %s -> 未捕获异常 耗时=%.1fms",
            request_id, request.method, request.url.path, cost_ms,
        )
        return JSONResponse(
            status_code=500,
            content={"error": "服务器内部错误", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )

    cost_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "[%s] %s %s -> %s 耗时=%.1fms",
        request_id, request.method, request.url.path, response.status_code, cost_ms,
    )
    response.headers["X-Request-ID"] = request_id   # 客户端拿得到，排查时能和日志对上
    return response


# ---------- 2. 统一异常出口 ----------
# 注册 Starlette 的 HTTPException：fastapi.HTTPException 是它的子类，
# 这样连 404 Not Found 也会走统一格式。
@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(request: Request, exc: StarletteHTTPException):
    """业务异常：缺参数、鉴权失败等，状态码由 raise 时指定。"""
    rid = getattr(request.state, "request_id", "-")
    logger.warning("[%s] 业务异常 %s: %s", rid, exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "request_id": rid},
        headers={"X-Request-ID": rid},
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    """Pydantic 校验失败：默认 422，这里整理成统一格式。"""
    rid = getattr(request.state, "request_id", "-")
    logger.warning("[%s] 参数校验失败: %s", rid, exc.errors())
    return JSONResponse(
        status_code=422,
        content={"error": "参数校验失败", "detail": exc.errors(), "request_id": rid},
        headers={"X-Request-ID": rid},
    )


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception):
    """最后一道防线，覆盖异常处理器自身或中间件外层的意外。"""
    rid = getattr(request.state, "request_id", "-")
    logger.exception("[%s] 未处理异常", rid)
    return JSONResponse(
        status_code=500,
        content={"error": "服务器内部错误", "request_id": rid},
        headers={"X-Request-ID": rid},
    )


# ---------- 3. 安全读取 JSON 请求体 ----------
async def read_json_or_400(request: Request) -> dict:
    """非法 JSON 抛 400 而不是 500 —— 这是客户端的问题，不是服务器的。"""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体不是合法 JSON") from None
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象")
    return data


# ---------- 4. 手动解析版：body / headers / query 全自己取 ----------
@app.post("/add_one")
async def add_one(request: Request):
    data = await read_json_or_400(request)
    number = data.get("number")
    token = request.headers.get("x-acs-dingtalk-access-token")
    operator_id = request.query_params.get("operatorId")

    logger.info(
        "[%s] 手取参数 body=%s token=%s operatorId=%s path=%s %s",
        request.state.request_id, data, token, operator_id,
        request.url.path, request.method,
    )

    # 校验失败一律 raise；return (dict, 400) 是 Flask 写法，FastAPI 会把元组当 JSON 返回
    if number is None:
        raise HTTPException(status_code=400, detail="缺少 number")
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        raise HTTPException(status_code=400, detail="number 必须是数字")
    if not token:
        raise HTTPException(status_code=401, detail="缺少 x-acs-dingtalk-access-token")

    return {"result": number + 1, "request_id": request.state.request_id}


# ---------- 5. Pydantic 版：声明式校验，框架自动 422 ----------
class AddOneRequest(BaseModel):
    number: float = Field(..., description="要加一的数字")


@app.post("/add_one_v2")
async def add_one_v2(payload: AddOneRequest):
    """和 /add_one 等价，但校验交给框架：少字段 / 类型不对 → 自动 422。"""
    return {"result": payload.number + 1}


@app.get("/hello")
async def hello(request: Request):
    name = request.query_params.get("name", "匿名")
    age = request.query_params.get("age", "0")
    token = request.headers.get("x-token")

    return {
        "msg": f"你好，{name}",
        "age": age,
        "token": token,
        "path": request.url.path,
        "method": request.method,
        "request_id": request.state.request_id,
    }


@app.get("/health")
async def health():
    """探测接口：只说明进程活着，不碰任何外部依赖。"""
    return {"status": "ok"}


@app.get("/boom")
async def boom():
    """故意抛异常：日志里能看到完整堆栈，客户端只会收到 500 + request_id。"""
    return {"value": 1 / 0}


if __name__ == "__main__":
    # 用 8001：8000 留给 06 的 /chat 服务，两边可以同时跑
    uvicorn.run(app, host="127.0.0.1", port=8001)
