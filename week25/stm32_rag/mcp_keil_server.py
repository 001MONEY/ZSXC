"""Keil MCP Server：把「工程操作 / 文件修改 / 编译 / 日志分析」暴露成 MCP 工具。

架构图里 Agent 通过 MCP 操作 Keil，而不是自己去拼命令行。这样：

- 任何支持 MCP 的客户端（Claude Desktop / VS Code / 自写 Agent）都能复用这套能力；
- 工程写回统一走 `KeilProject.save()`，不会出现 BOM 导致 UV4 读不了的问题；
- 编译统一走 `KeilBuilder`，退出码、日志、结构化诊断一次到位。

启动方式：

    python -m stm32_rag.mcp_keil_server --workspace D:\\path\\to\\project
    python -m stm32_rag.mcp_keil_server --workspace ... --transport sse --port 8765

环境变量：
    STM32_WORKSPACE     默认工作区（未传 --workspace 时使用）
    STM32_UVPROJX       直接指定 .uvprojx
    STM32_UV4           UV4.exe 路径
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .core.config import DEFAULT_DEVICE, WORKSPACE_ROOT
from .build.keil import KeilBuilder, suggest_fixes
from .gen.project import KeilProject, find_project
from .gen.scaffold import scaffold_project

mcp = FastMCP("keil-mdk")

_STATE: dict[str, object] = {"workspace": None, "project": None, "build": None}


def _workspace() -> Path:
    if _STATE["workspace"] is None:
        root = os.getenv("STM32_WORKSPACE") or str(WORKSPACE_ROOT)
        _STATE["workspace"] = Path(root).expanduser().resolve()
    return _STATE["workspace"]  # type: ignore[return-value]


def _project(create: bool = False) -> KeilProject:
    cached = _STATE.get("project")
    if isinstance(cached, KeilProject):
        return cached
    explicit = os.getenv("STM32_UVPROJX")
    path = Path(explicit).expanduser() if explicit else find_project(_workspace())
    if path is None:
        if not create:
            raise FileNotFoundError(f"{_workspace()} 下没有 .uvprojx（可先调用 keil_scaffold）")
        result = scaffold_project(_workspace(), device=DEFAULT_DEVICE, name="stm32_app")
        path = Path(result["project"])
    project = KeilProject(path)
    _STATE["project"] = project
    return project


def _resolve(rel: str) -> Path:
    workspace = _workspace()
    path = (workspace / rel.replace("\\", "/")).resolve()
    if path != workspace and workspace not in path.parents:
        raise ValueError(f"路径越界：{rel}")
    return path


# ---------------------------------------------------------------- 工程

@mcp.tool()
def keil_scaffold(name: str = "stm32_app", device: str = DEFAULT_DEVICE) -> str:
    """在工作区生成一个可编译的 Keil 工程（器件默认 STM32F103RB）。"""
    result = scaffold_project(_workspace(), device=device, name=name)
    _STATE["project"] = None
    return f"已生成工程：{result['project']}（{result['device']}）"


@mcp.tool()
def keil_project_info() -> str:
    """查看工程器件、分组、文件、宏定义、头文件路径与缺失文件。"""
    return _project().describe()


@mcp.tool()
def keil_list_sources(group: str = "") -> str:
    """列出工程里的源文件；可按分组过滤（如 Startup / Application/User）。"""
    files = _project().files()
    if group:
        files = [item for item in files if group.lower() in item.group.lower()]
    if not files:
        return "没有匹配的文件"
    return "\n".join(f"[{item.group}] {item.name}  →  {item.path}" for item in files)


@mcp.tool()
def keil_read_file(path: str, limit: int = 6000) -> str:
    """读取工作区内文件（相对工作区根）。"""
    target = _resolve(path)
    if not target.is_file():
        return f"文件不存在：{path}"
    text = target.read_text(encoding="utf-8", errors="replace")
    return text[:limit] + ("\n…（已截断）" if len(text) > limit else "")


@mcp.tool()
def keil_write_file(path: str, content: str) -> str:
    """写入工作区内文件（整文件覆盖，注意先读再写）。"""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"已写入 {path}（{len(content)} 字符）"


@mcp.tool()
def keil_add_source(path: str, group: str = "Application/User") -> str:
    """把源文件加入工程分组（path 相对工作区根，如 Core/Src/app.c）。"""
    project = _project()
    added = project.add_file(path, group=group)
    if added:
        project.save()
        return f"已加入工程：[{group}] {path}"
    return f"已存在：{path}"


@mcp.tool()
def keil_set_defines(defines: str) -> str:
    """设置工程宏定义（逗号分隔，会整体替换，如 USE_HAL_DRIVER,STM32F103xB）。"""
    project = _project()
    values = [item.strip() for item in defines.split(",") if item.strip()]
    project.set_defines(values)
    project.save()
    return f"宏定义已设为：{', '.join(values)}"


@mcp.tool()
def keil_add_include_paths(paths: str) -> str:
    """追加头文件搜索路径（分号分隔，相对 MDK-ARM，如 ..\\Core\\Inc）。"""
    project = _project()
    added = project.add_include_path(*[item for item in paths.split(";") if item.strip()])
    project.save()
    return f"新增路径：{'; '.join(added)}" if added else "路径已存在"


# ---------------------------------------------------------------- 编译

@mcp.tool()
def keil_build(clean: bool = False) -> str:
    """编译当前工程。clean=True 时先删中间产物做全量编译。"""
    project = _project()
    result = KeilBuilder().build(project.path, clean=clean)
    _STATE["build"] = result
    text = result.summary()
    if not result.ok:
        fixes = suggest_fixes(result.errors)
        if fixes:
            text += "\n建议修复：\n" + "\n".join(f"- {fix.action} {fix.value}（{fix.reason}）" for fix in fixes)
    return text


@mcp.tool()
def keil_last_errors() -> str:
    """查看最近一次编译的错误/警告（JSON）。"""
    result = _STATE.get("build")
    if result is None:
        return "还没有编译记录，先调用 keil_build"
    import json

    return json.dumps(result.as_dict(), ensure_ascii=False, indent=2)  # type: ignore[union-attr]


@mcp.tool()
def keil_build_log(tail: int = 80) -> str:
    """查看最近一次编译日志的末尾若干行。"""
    result = _STATE.get("build")
    if result is None or not getattr(result, "log_path", None):
        return "还没有编译日志"
    path = Path(result.log_path)  # type: ignore[union-attr]
    if not path.is_file():
        return f"日志文件不存在：{path}"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-tail:])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stm32_rag.mcp_keil_server", description="Keil MCP Server")
    parser.add_argument("--workspace", help="工程工作区（默认 STM32_WORKSPACE 或 week25/workspaces）")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    if args.workspace:
        _STATE["workspace"] = Path(args.workspace).expanduser().resolve()

    if args.transport == "sse":
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
