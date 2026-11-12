# MinerU 4.2 本地安装与使用（week25）

把 PDF / 图片 / Office 文档转成 Markdown（含公式、表格、版面结构），用于给 RAG 做高质量切块。

## 安装位置

| 项目 | 值 |
|------|-----|
| Python 环境 | `D:\project\step3\llm`（conda，Python 3.11.16 + torch 2.10.0+cu128） |
| 版本 | `mineru 4.2.0`（安装时 `torch` 未被替换） |
| 命令目录 | `D:\project\step3\llm\Scripts\`（不在 PATH，需写全路径或自行加入 PATH） |
| 本地主目录 | `C:\Users\13744\.mineru`（配置 `config.yaml`、缓存 `doclib\parsed`、日志 `logs\doclib.log`） |
| 解析后端 | 本地 managed，档位 `basic`（适配 6 GB 显存的 RTX 3060） |
| 模型来源 | ModelScope `OpenDataLab/MinerU-4_models_torch`（已下载 12 个文件） |

## 一次性配置（已执行）

```powershell
$env:MINERU_MODEL_SOURCE = 'modelscope'
& D:\project\step3\llm\Scripts\mineru-kit.exe models download --tier basic
& D:\project\step3\llm\Scripts\mineru.exe config set parse_server.local.managed_tier basic
& D:\project\step3\llm\Scripts\mineru.exe config set parse_server.local.mode managed
```

查看配置：`mineru config show`（关键项：`parse_server.local.mode = managed`）

## 日常使用

```powershell
# 1) 启动常驻服务（必须，parse 依赖它）
Start-Process -FilePath 'D:\project\step3\llm\Scripts\mineru.exe' -ArgumentList 'server','start' -WindowStyle Hidden

# 2) 查看状态（本地 Parse Server 需显示“健康”）
& D:\project\step3\llm\Scripts\mineru.exe server status

# 3) 解析（-o 是【输出文件】不是目录；首次解析要先等模型加载）
& D:\project\step3\llm\Scripts\mineru.exe parse "D:\path\to\doc.pdf" -o "D:\out\doc.md" --tier basic --wait 300 -v

# 4) 停止服务
& D:\project\step3\llm\Scripts\mineru.exe server stop
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--tier flash/basic/standard/advanced` | 解析档位，`basic` 适合 6 GB 显存 |
| `--pages 1-5,8` / `--pages r3-r1` / `all` | 指定页码（默认前 10 页） |
| `--format markdown` | 输出格式（当前仅 markdown） |
| `--force` | 忽略缓存重新解析 |
| `--remote` | 改用 mineru.net 官方远端服务（需 API Key） |
| `--no-wait` | 只入队不等待结果 |

其他命令：`mineru read`（按页/定位读已解析内容）、`mineru search`（检索解析结果）、`mineru scan`（扫描目录入库）、`mineru server`。

## 已验证的示例

```powershell
& D:\project\step3\llm\Scripts\mineru.exe parse d:\project\step1\week25\_demo_two_pages.pdf -o d:\project\step1\week25\mineru_out\demo_two_pages.md --tier basic -v --wait 300
```

输出（2 页，约 30–60 s）：

```markdown
<!-- page 1 of 2 -->

## Page 1: Emei Mountain guide
Highest peak: Wanfoding, 3099 m

<!-- page 2 of 2 -->
Page 2: Buddhist etiquette
```

## 注意事项

1. **必须先起服务**：`mineru parse` 走本地 API（`127.0.0.1:15980`），服务没起会报“本地 mineru 服务未运行”
2. **服务启动后要等模型加载完成**（一般 30–90 秒），`server status` 里本地目标显示“健康”再解析，否则报 `Local parse-server is not ready`
3. `-o` 是文件路径；要批量输出到目录请自己拼文件名
4. 解析结果同时缓存在 `C:\Users\13744\.mineru\doclib\parsed\...`，可用 `mineru read` / `mineru search` 复用
5. 遥测默认开启，介意可执行 `mineru telemetry disable`，或启动前设 `MINERU_TELEMETRY_DISABLED=1`
6. 中文控制台输出可能乱码，先执行 `[Console]::OutputEncoding = [Text.Encoding]::UTF8`

## 与 week25 RAG 的衔接

```
PDF → mineru parse → Markdown（带页标记） → 现有 chunker.py 切块 → text-embedding-v4 → Chroma / Milvus
```

相比 `pypdf` 直接抽文本，MinerU 能保留标题层级、表格（HTML）、公式（LaTeX）和图片位置，切块质量明显更好；`<!-- page N of M -->` 标记还能给 chunk 补页码 metadata，便于引用定位。
