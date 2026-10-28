# LangChain 学习环境（week23）

使用 [uv](https://docs.astral.sh/uv/) 管理的 Python 3.11 虚拟环境，用于 LangChain 学习与实验。

## 目录结构

```
week23/
├── .venv/                 # uv 创建的虚拟环境（已 gitignore）
├── .python-version        # 锁定 Python 3.11
├── pyproject.toml         # 依赖声明（uv add 自动维护）
├── uv.lock                # 精确锁文件（务必提交）
├── .env.example           # 环境变量模板 → 复制为 .env
├── 01_quickstart.ipynb    # 冒烟测试：版本检查 + 首个 LCEL 链
├── 20261027/              # 课堂代码（已做本地化改造）
└── deploy/                # 已有的 vLLM 部署笔记
```

## 常用命令

```powershell
# 激活虚拟环境（PowerShell）
.venv\Scripts\Activate.ps1

# 不激活、直接用 uv 运行脚本
uv run python script.py

# 新增 / 删除依赖
uv add langgraph
uv remove langchain-chroma

# 按 uv.lock 精确还原环境（换机器 / 重装）
uv sync

# 查看已装包
uv pip list
```

## 依赖说明

| 包 | 用途 |
|---|---|
| `langchain` | 核心框架（LCEL、Agent、工具、链） |
| `langchain-core` | 由 `langchain` 自动带入，抽象基类（Runnable / Message / Tool） |
| `langchain-openai` | OpenAI 兼容接口，可对接 DeepSeek、通义、智谱等云端模型 |
| `langchain-ollama` | 本地 Ollama 模型（本机已装 Ollama，默认 `127.0.0.1:11434`） |
| `langchain-community` | 社区集成（文档加载器、工具等） |
| `langchain-text-splitters` | 文本切分，RAG 必备 |
| `langchain-chroma` | Chroma 向量库，做 RAG |
| `python-dotenv` | 读取 `.env` 中的 API Key |
| `ipykernel` | 让 VS Code Notebook 能选到本环境的 kernel |

实际安装版本（2026-10-27，共 137 个包）：`langchain 1.4.7`、`langchain-core 1.7.0`、`langchain-openai 1.7.3`、`langchain-ollama 1.1.0`、`langchain-community 0.4.2`、`langchain-text-splitters 1.1.3`、`langchain-chroma 1.1.0`（自带 `chromadb 1.5.9`）。
`langgraph 1.2.18` 已作为 `langchain 1.x` 的依赖自动安装，无需再 `uv add`。

## 配置模型

1. 复制 `.env.example` 为 `.env`
2. 二选一：
   - **云端**：填 `OPENAI_API_KEY` + `OPENAI_BASE_URL`（DeepSeek 用 `https://api.deepseek.com/v1`）
   - **本地**：先 `ollama pull qwen3:8b`，再填 `OLLAMA_MODEL`

> `.env` 已被 `.gitignore` 忽略，不要提交密钥。

## 在 VS Code Notebook 中选内核

本环境已注册为 Jupyter kernelspec：

- 内核名：`week23-langchain`
- 显示名：`Python 3.11 (week23 LangChain)`

打开任意 `.ipynb` → 右上角选择内核 → 选 `Python 3.11 (week23 LangChain)`；
或者选 **Python Environments** → `week23/.venv`（`Python 3.11.9`）。

重新注册（例如重建 `.venv` 之后）：

```powershell
uv run python -m ipykernel install --user --name "week23-langchain" --display-name "Python 3.11 (week23 LangChain)"
```

## 安装加速（国内网络）

默认 PyPI 在国内很慢（实测只有几十 KB/s）。已验证可用的镜像：

```powershell
# 腾讯云（已同步 langchain-openai 1.7.3，推荐）
$env:UV_DEFAULT_INDEX="https://mirrors.cloud.tencent.com/pypi/simple"
uv sync
```

踩坑记录：

- 清华源 `pypi.tuna.tsinghua.edu.cn` 对部分 wheel 返回 **403**（如 `ipykernel-7.4.0`）
- 阿里云源 `mirrors.aliyun.com/pypi/simple` **索引滞后**，缺 `langchain-openai 1.7.3`，会导致解析失败
- USTC 源对本机直接 403

如果希望永久生效，可在 `pyproject.toml` 加：

```toml
[[tool.uv.index]]
url = "https://mirrors.cloud.tencent.com/pypi/simple"
default = true
```

## 课堂代码（`20261027/`）

课堂原始代码把 API Key、老师机器的局域网 IP（`192.168.1.105`）写死在各文件里，现已统一改造。

| 文件 | 内容 |
|---|---|
| `lab_config.py` | **共用配置**：从 `week23/.env` 读凭据，按 `LLM_PROVIDER` 创建模型 |
| `01_openai_sdk_stream.py` | 原生 `openai` SDK 流式，分离 `reasoning_content` 与正文 |
| `02_model_init_three_ways.py` | `ChatOpenAI` / `ChatOllama` / `init_chat_model` 三种方式对比 |
| `03_json_output.py` | 长文本 + System 约束让模型输出 JSON 风格 |
| `04_message_types.py` | `SystemMessage` / `HumanMessage` / `AIMessage` 搭消息列表 |
| `05_multi_turn_cli.py` | 命令行多轮对话 |
| `06_chat_server_fastapi.py` | FastAPI 接口 `POST /chat`（主模型千问，失败重试 3 次后降级到 DeepSeek） |
| `07_chat_client.py` | 调 06 起的服务的多轮客户端 |
| `08_fastapi_request_basics.py` | 手动解析 Request 的 body / headers / query，含请求日志中间件与统一异常出口 |
| `09_requests_params.py` | `requests` 的 `params` / `headers` / `json` 三处参数 |

切换到千问，改 `.env`：

```dotenv
LLM_PROVIDER=qwen
QWEN_API_KEY=sk-你的百炼Key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
# 课堂专用网关则是：
# QWEN_BASE_URL=https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1
# QWEN_MODEL=qwen3.8-flash
```

运行方式（必须先进入该目录，`lab_config` 才导入得到）：

```powershell
cd 20261027
python lab_config.py    # 看当前用的是哪个模型（Key 会打码）
python 02_model_init_three_ways.py   # 跑任意一个脚本
```

端口约定：`06`/`07` 用 **8000**（`/chat` 服务），`08`/`09` 用 **8001**（`/add_one`、`/hello`），两组可同时运行。
服务端是常驻进程，必须**一个终端跑服务端、另一个终端跑客户端**，否则后一个命令只会在队列里等着。

### 08 的容错与日志设计

| 机制 | 作用 |
|---|---|
| `@app.middleware("http")` | 每个请求生成 8 位 `request-id`，记录方法/路径/状态码/耗时，并在响应头回写 `X-Request-ID` |
| `read_json_or_400()` | 非法 JSON / 非对象请求体 → 400，而不是让框架恼羞成怒抰 500 |
| `HTTPException` | 缺参数 400、缺 token 401；**不能**写 `return {"error": ...}, 400`（FastAPI 会当 JSON 返回并给 200） |
| `RequestValidationError` | Pydantic 模型校验失败 → 422，统一成 `{error, detail, request_id}` |
| `StarletteHTTPException` | 连 404 也走统一格式 |
| `Exception` + 中间件兜底 | 未捕获异常：客户端只看到 500 + `request_id`，**堆栈只进日志** |

调试用接口：`GET /health`（存活探测）、`GET /boom`（故意 1/0，看日志与响应的差异）、`POST /add_one_v2`（Pydantic 自动校验版）。

改造中修掉的 bug：

- `06_chat_server_fastapi`：`response.status_code`（`AIMessage` 没这个属性）、缺 `try/except` 导致备用模型永不触发、重试耗尽后返回 `null`
- `07_chat_client`：`requests.post(url, json=data)` 用错变量，形参 `messages` 成了摆设
- `08_fastapi_request_basics`：`if not number` 会把 `number=0` 误判成缺参数
- 原 `test1~7` 按学习顺序重命名为语义化文件名（`teste6.py` 的拼写错误一并修正）

### 输出「讲一半就停」怎么排查

`qwen3.8-flash` 是**思考型模型**：回答前先产出一段 `reasoning_content`。如果 `OPENAI_MAX_TOKENS` 不够大，思考会把额度花光、正文为空。

```python
r = get_model().invoke([{"role": "user", "content": "..."}])
r.response_metadata["finish_reason"]        # 'length' → 被截断
r.usage_metadata["output_token_details"]    # {'reasoning': 512} → 全烧在思考上
```

解决：`.env` 里设 `QWEN_ENABLE_THINKING=false`（默认已关），必要时再调大 `OPENAI_MAX_TOKENS`。

实测同一个问题：

| 配置 | 正文字数 | 输出 token |
|---|---|---|
| 关思考，max_tokens=512 | 597 | 296 |
| 开思考，max_tokens=2048 | 422 | 887（其中 658 是推理） |

## 建议学习路径

1. **Model I/O**：`ChatOpenAI` / `ChatOllama`、Prompt Template、Output Parser
2. **LCEL**：`prompt | model | parser` 管道与 `RunnablePassthrough`、`RunnableParallel`
3. **Memory / 多轮对话**：`MessagesPlaceholder` + 消息列表管理
4. **RAG**：Loader → TextSplitter → Embeddings → Chroma → Retriever → 生成
5. **Tools / Agent**：`@tool`、`bind_tools`、`create_agent`（LangChain 1.x 的 Agent 基于 LangGraph）
6. **LangGraph**：已随 `langchain 1.x` 安装，直接 `from langgraph.graph import StateGraph`，做状态机式的可控编排
