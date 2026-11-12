# Milvus Standalone 本地安装（Docker Compose）

按官方 Windows 文档 <https://milvus.io/docs/zh/install_standalone-windows.md> 在本机运行 Milvus 单机版，
使用官方 `v3.0.5` 的 compose 配置，用于 RAG 学习与实验。

## 组成

| 服务 | 镜像 | 作用 | 端口 |
|------|------|------|------|
| `milvus-standalone` | `milvusdb/milvus:v3.0.5` | Milvus 本体（gRPC） | 19530（gRPC）、9091（metrics / healthz / WebUI） |
| `milvus-volume-init` | 同上（一次性任务） | 修正 `volumes/milvus` 的属主 | 无 |
| `milvus-etcd` | `quay.io/coreos/etcd:v3.5.25` | 元数据存储 | 仅容器内网 2379 |
| `milvus-minio` | `milvusdb/minio:RELEASE.2024-12-18T13-15-44Z` | 对象存储（段文件） | 9000（API）、9001（控制台） |
| `milvus-attu`（可选） | `zilliz/attu:v3.0.1` | Web 图形控制台 | 8000 → 3000 |

Milvus 3.0 默认使用本地 WAL（Woodpecker），不需要外部 Pulsar/Kafka。
数据持久化目录：`.\volumes\{etcd,minio,milvus}`（首次启动自动创建）。

## 常用命令

```powershell
cd d:\project\step1\week25\milvus

.\manage.ps1 start      # 启动（docker compose up -d）
.\manage.ps1 status     # 查看容器 + healthz
.\manage.ps1 logs       # 跟踪 standalone 日志
.\manage.ps1 webui      # 打开 Milvus 内置 WebUI（http://127.0.0.1:9091/webui）
.\manage.ps1 gui        # 启动 Milvus + Attu 控制台（http://localhost:8000）
.\manage.ps1 stop       # 停止（保留数据）
.\manage.ps1 reset      # 停止并清空 volumes（危险）
```

等价的原生命令：

```powershell
docker compose -f d:\project\step1\week25\milvus\docker-compose.yml up -d
docker compose -f d:\project\step1\week25\milvus\docker-compose.yml ps
docker compose -f d:\project\step1\week25\milvus\docker-compose.yml down
```

> 注意：必须在**本目录**执行 `docker compose`，或像上面一样用 `-f` 指定 compose 文件绝对路径；
> 否则 compose 会把上一级 `week25` 当作项目根，误操作其他 compose 项目。

## Python 客户端

已在本机 `d:\project\step1\week23\.venv`（Python 3.11.9）中安装 `pymilvus==3.0.3`。
其他环境安装：`pip install "pymilvus>=3.0,<3.1"`

自检脚本：

```powershell
& D:\project\step1\week23\.venv\Scripts\python.exe d:\project\step1\week25\milvus\test_milvus.py
```

预期输出：服务端版本、创建 `demo_rag` 集合、写入 3 条数据、Top2 检索结果、`[OK] Milvus 安装成功`。

> 注意：Milvus 3.0 下刚 `insert` 的数据需要 `client.flush(collection)`（或等待几秒）才稳定可检索。

## 最小代码示例

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://127.0.0.1:19530")

schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=768)
schema.add_field("text", DataType.VARCHAR, max_length=1024)

index_params = client.prepare_index_params()
index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

client.create_collection("rag_docs", schema=schema, index_params=index_params)
client.insert("rag_docs", [{"id": 1, "vector": [0.1] * 768, "text": "示例"}])
client.load_collection("rag_docs")
res = client.search("rag_docs", data=[[0.1] * 768], limit=3, output_fields=["text"])
```

与 LangChain 结合（可选）：

```powershell
pip install langchain-milvus
```

```python
from langchain_milvus import Milvus
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": "http://127.0.0.1:19530"},
    collection_name="rag_docs",
)
```

## 端口速查

- `19530`：客户端连接地址（如 `MilvusClient(uri="http://127.0.0.1:19530")`）
- `9091`：健康检查 `http://127.0.0.1:9091/healthz`、Prometheus 指标、内置 WebUI `http://127.0.0.1:9091/webui`
- `9001`：MinIO 控制台（账号/密码 `minioadmin` / `minioadmin`）
- `8000`：Attu 控制台（`manage.ps1 gui` 启动）
  - 首次打开会要求**创建 Attu 管理员账号**（用户名/邮箱 + 密码，8–128 位且至少含大写/小写/数字/符号中的三类）
  - 注册后进入 `/connect`，Milvus 地址填 `milvus-standalone`、端口 `19530`（容器内访问，无需鉴权）

## 排障

| 现象 | 处理 |
|------|------|
| `docker` 连接失败 `dockerDesktopLinuxEngine` | 先启动 Docker Desktop，等待鲸鱼图标变绿 |
| 拉镜像慢/超时 | 已配置镜像加速；`quay.io` 不受加速影响，可 `docker pull quay.mirrors.ustc.edu.cn/coreos/etcd:v3.5.25` 后 `docker tag` 改名 |
| Docker 自带拉取很慢（几 KB/s） | 本机开了 Clash TUN，镜像域名被代理解析；可改用 `docker pull docker.1ms.run/milvusdb/milvus:v3.0.5` 再 `docker tag` 回官方名，或临时关闭 TUN |
| standalone 反复重启 | `docker compose logs standalone` 查看；多为 `vm.max_map_count` 过小（WSL2 需 ≥ 262144）或内存不足（建议 ≥ 8 GB 空闲） |
| 端口被占用 | 检查 `19530/9091/9000/9001` 是否被其他服务监听 |
| 想彻底重来 | `.\manage.ps1 reset` 删除 `.\volumes` 后重新 `start` |

## 版本说明

- 服务端：`v3.0.5`（官方 Windows 文档当前引用的版本，`milvus-standalone` + `minio` + `etcd` + 一次性 `milvus-volume-init`）
- 客户端：`pymilvus 3.0.3`
- 本目录另有历史配置 `docker-compose.v2.6.yml`（v2.6.27，如需回退可用它 + `pymilvus 2.6.x`）
