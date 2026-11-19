# Neo4j 本地部署（week25/neo4j）

Neo4j 5.26 Community（LTS）+ APOC，Docker Compose 方式，与 `..\milvus` 同一套路。

## 端口与账号

| 用途 | 地址 |
|------|------|
| Neo4j Browser（Web UI） | http://localhost:7474 |
| Bolt 驱动 | `bolt://localhost:7687` |
| 账号 | `neo4j` / `neo4j123456` |

## 启停

```powershell
# 在 d:\project\step1\week25\neo4j 下
.\manage.ps1 start     # 启动（首次会下载 APOC 插件）
.\manage.ps1 status    # 容器状态
.\manage.ps1 logs      # 最近 100 行日志
.\manage.ps1 shell     # 进入 cypher-shell
.\manage.ps1 browser   # 打开浏览器
.\manage.ps1 stop      # 停止
.\manage.ps1 reset     # 删除全部数据（需输入 YES）
```

也可以直接 `docker compose -f d:\project\step1\week25\neo4j\docker-compose.yml up -d`
（一定要带 `-f`：终端工具会丢掉 `Set-Location`，否则会在 week25 根目录执行）。

## 数据与配置

- 数据：`neo4j/volumes/neo4j/data`（挂到容器 `/data`），日志：`volumes/neo4j/logs`
- 导入目录：`neo4j/import` → 容器 `/import`（`LOAD CSV` 用 `file:///xxx.csv`）
- 改密码：编辑 `docker-compose.yml` 里的 `NEO4J_AUTH` 后 `.\manage.ps1 restart`
  （已初始化过的库不会因改 `NEO4J_AUTH` 而改密码，需 `reset` 或 `ALTER USER`）

## Python 驱动

```powershell
uv pip install --python d:\project\step1\week23\.venv\Scripts\python.exe neo4j
d:\project\step1\week23\.venv\Scripts\python.exe neo4j\test_neo4j.py
```

`test_neo4j.py` 会依次验证：连接/版本、APOC 版本、建图查询、向量索引（GraphRAG 需要）、清理。

## 常见问题

- **拉镜像（2026-11-19 实测）**：
  - Clash 开着：Docker Hub 直连被 TUN fake-ip 挡死（0 B/s），`docker.1ms.run` 仅 ~68 KB/s
  - Clash 关闭：`registry-1.docker.io` 直接连接被重置，`docker.1ms.run` ~170 KB/s 且会掉速
  - **`docker.m.daocloud.io` 23 MB/s，934MB 几十秒拉完 → 首选**：
    ```
    docker pull docker.m.daocloud.io/library/neo4j:5.26-community
    docker tag  docker.m.daocloud.io/library/neo4j:5.26-community neo4j:5.26-community
    ```
  - 镜像站地址必须是 `library/<image>:<tag>`（写成 `.../neo4j/neo4j:tag` 会 404）
- **首次启动慢**：Windows 绑定挂载 `./volumes/neo4j/data` 上建库要 1 分钟左右
  （日志停在 “Neo4j 5.26.33” 之后是正常的，看 `volumes/neo4j/logs/debug.log` 有没有继续写）。
  健康检查要到 `Up (healthy)` 才算好；`docker ps` 显示 `health: starting` 时连 Bolt 会报
  `incomplete handshake response`。
- **APOC 不用联网**：镜像自带 `labs/apoc-*-core.jar`，`NEO4J_PLUGINS=["apoc"]` 只是本地拷贝。
- **端口冲突**：7474 / 7687 被占用时改宿主机侧端口即可。
