"""Agent 工具集：把 RAG / 工程分析 / 代码生成 / 编译 都包装成 LangChain Tool。

设计原则：**工具做确定性的事，模型只做决策**。

- `generate_application` 走 `codegen.plan_application()`，模型给规格，不手写 C 代码；
- `build_project` 走 `keil.KeilBuilder`，模型看结构化诊断；
- `write_source` 只允许写工作区内文件，防止模型把别处改坏。

这样即使模型写错规格，也只会得到一条明确的校验错误，而不是一个编译不过的工程。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from ..gen.codegen import (
    AppSpec,
    GpioSpec,
    LoopSpec,
    PwmSpec,
    TimerSpec,
    UartSpec,
    plan_application,
)
from ..core.config import DEFAULT_DEVICE, DEFAULT_HSE, DEFAULT_SYSCLK, WORKSPACE_ROOT
from ..build.keil import BuildResult, KeilBuilder, suggest_fixes
from ..gen.mcu import USART_PINS, parse_pin, resolve_device
from ..build.pipeline import _sync_project_config, apply_quick_fix
from ..gen.project import KeilProject, find_project
from ..gen.scaffold import scaffold_project


@dataclass
class WorkspaceSession:
    """一次开发任务的工作区 + 工程 + 构建状态。"""

    workspace: Path
    project_path: Path | None = None
    device: str = DEFAULT_DEVICE
    sysclk: int = DEFAULT_SYSCLK
    hse: int = DEFAULT_HSE
    build: BuildResult | None = None
    spec: AppSpec | None = None
    actions: list[str] = field(default_factory=list)
    _project: KeilProject | None = None
    _graph: object | None = field(default=None, repr=False)
    _doc_graph: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """CubeMX 工程里优先用 .ioc 里的真实器件型号。"""
        try:
            from ..gen.cubemx import find_ioc, parse_ioc

            ioc = find_ioc(self.workspace)
            if ioc is not None:
                info = parse_ioc(ioc)
                if info.mcu:
                    self.device = info.mcu
        except Exception:
            pass

    # ------------------------------------------------------------ 工程
    @property
    def project(self) -> KeilProject | None:
        if self._project is not None:
            return self._project
        path = self.project_path or find_project(self.workspace)
        if path is None:
            return None
        self._project = KeilProject(path)
        return self._project

    def ensure_project(self) -> KeilProject:
        project = self.project
        if project is not None:
            return project
        scaffold = scaffold_project(
            self.workspace,
            device=self.device,
            name="stm32_app",
            modules=("tim", "uart"),
            hse=self.hse,
        )
        self.project_path = Path(scaffold["project"])
        self._project = KeilProject(self.project_path)
        self.actions.append(f"生成工程 {self.project_path.name}")
        return self._project

    # ------------------------------------------------------------ 路径
    def resolve(self, rel: str) -> Path:
        path = (self.workspace / rel.replace("\\", "/")).resolve()
        root = self.workspace.resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"路径越界：{rel}")
        return path

    # ------------------------------------------------------------ 动作
    def overview(self) -> str:
        project = self.project
        if project is None:
            return f"工作区 {self.workspace} 里还没有 Keil 工程（可用 generate_application 自动创建）"
        lines = [project.describe()]
        try:
            from ..gen.cubemx import find_ioc, parse_ioc

            ioc = find_ioc(self.workspace)
            if ioc is not None:
                lines.append("")
                lines.append(parse_ioc(ioc).describe())
        except Exception as exc:  # .ioc 解析失败不影响工程信息
            lines.append(f"（.ioc 解析失败：{exc}）")
        return "\n".join(lines)

    def read(self, rel: str, limit: int = 4000) -> str:
        path = self.resolve(rel)
        if not path.is_file():
            return f"文件不存在：{rel}"
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[:limit] + ("\n…（已截断）" if len(text) > limit else "")

    def write(self, rel: str, content: str) -> str:
        path = self.resolve(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        before = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        path.write_text(content, encoding="utf-8")
        changed = "新建" if not before else ("已更新" if before != content else "内容未变")
        self.actions.append(f"{changed} {rel}")
        return f"{changed}：{rel}（{len(content)} 字符）"

    def generate(self, payload: str) -> str:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            return f"规格 JSON 解析失败：{exc}。示例：{{\"pwm\":[{{\"pin\":\"PA8\",\"freq_hz\":1000000,\"duty\":0.5}}]}}"
        spec = spec_from_dict(data, self)
        self.spec = spec
        plan = plan_application(spec)
        if not plan.ok:
            return "规格校验未通过：\n" + "\n".join(f"- {item}" for item in plan.errors)

        project = self.ensure_project()
        changed: list[str] = []

        # CubeMX 工程：只往 USER CODE 段插入，不覆盖 CubeMX 生成的代码
        from ..gen.cubemx import apply_to_project, find_ioc

        main_c = self.workspace / "Core" / "Src" / "main.c"
        cubemx_mode = bool(
            find_ioc(self.workspace)
            and main_c.is_file()
            and "USER CODE BEGIN" in main_c.read_text(encoding="utf-8", errors="replace")
        )
        if cubemx_mode:
            patch = apply_to_project(self.workspace, spec, project)
            if not patch.ok:
                return "CubeMX 增量修改失败：\n" + "\n".join(f"- {item}" for item in patch.errors)
            sync = _sync_project_config(self.workspace, project, patch.modules or plan.modules)
            project.save()
            self.actions.append(f"CubeMX 增量修改：{', '.join(patch.changed_files) or '无变化'}")
            merged = patch.spec or spec
            lines = [f"已在 CubeMX 工程上增量修改（{merged.device.name}，SYSCLK {merged.sysclk / 1e6:g} MHz）"]
            if patch.changed_files:
                lines.append("修改文件：" + "、".join(patch.changed_files))
            if patch.applied:
                lines.append("已应用：")
                lines.extend(f"- {item}" for item in patch.applied)
            if patch.skipped:
                lines.append("已跳过（CubeMX 已配置）：")
                lines.extend(f"- {item}" for item in patch.skipped)
            if sync:
                lines.append("工程配置：" + "；".join(sync))
            if patch.warnings:
                lines.extend(f"提醒：{item}" for item in patch.warnings)
            checks = patch.checks or plan.checks
            if checks:
                lines.append("计算核对：")
                lines.extend(f"- {item}" for item in checks)
            return "\n".join(lines)

        for rel, content in plan.files.items():
            path = self.resolve(rel)
            before = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
            if before.strip() != content.strip():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                changed.append(rel)
        for rel in ("Core/Src/main.c", "Core/Src/stm32f1xx_it.c", "Core/Src/system_stm32f1xx.c"):
            project.add_file(rel, group="Application/User" if "main" in rel or "it" in rel else "CMSIS")
        startup = self.workspace / "Core" / "Startup" / spec.device.startup
        if startup.is_file():
            project.add_file(f"Core/Startup/{startup.name}", group="Startup")
        sync = _sync_project_config(self.workspace, project, plan.modules)
        project.save()
        self.actions.append(f"生成代码：{', '.join(changed) or '无变化'}")

        lines = [f"已生成（{spec.device.name}，SYSCLK {spec.sysclk / 1e6:g} MHz）"]
        lines.append("写入文件：" + ("、".join(changed) if changed else "无变化"))
        if sync:
            lines.append("工程配置：" + "；".join(sync))
        if plan.checks:
            lines.append("计算核对：")
            lines.extend(f"- {item}" for item in plan.checks)
        if plan.warnings:
            lines.append("提醒：")
            lines.extend(f"- {item}" for item in plan.warnings)
        if plan.wiring:
            lines.append("接线：")
            lines.extend(f"- {item}" for item in plan.wiring)
        return "\n".join(lines)

    def build_project(self, clean: bool = False) -> BuildResult:
        project = self.ensure_project()
        builder = KeilBuilder()
        result = builder.build(project.path, clean=clean)
        self.build = result
        self.actions.append(f"编译：{'通过' if result.ok else '失败'}（{result.program_size or '无体积信息'}）")
        return result

    def build_report(self, clean: bool = False) -> str:
        result = self.build_project(clean=clean)
        text = result.summary()
        if not result.ok:
            fixes = suggest_fixes(result.errors)
            if fixes:
                text += "\n可自动修复项：\n" + "\n".join(f"- {fix.action} {fix.value}（{fix.reason}）" for fix in fixes)
            if result.log_path:
                text += f"\n完整日志：{result.log_path}"
        return text

    def auto_fix(self) -> str:
        """按最近一次构建错误执行工程配置修复并重新编译。"""
        if self.build is None or self.build.ok:
            return "没有待修复的构建错误"
        project = self.ensure_project()
        notes: list[str] = []
        for fix in suggest_fixes(self.build.errors):
            if fix.action == "manual":
                notes.append(f"跳过人工项：{fix.reason}")
                continue
            notes.append(apply_quick_fix(self.workspace, project, fix))
        project.save()
        rebuilt = self.build_report()
        return "\n".join(notes + ["", rebuilt])

    # ------------------------------------------------------------ 工具
    def tools(self) -> list[BaseTool]:
        session = self

        @tool
        def stm32_knowledge_search(query: str, sources: str = "auto", k: int = 6) -> str:
            """检索 STM32F103 知识库（RM0008 寄存器手册 / UM1850 HAL 说明 / HAL 源码 / 例程 / 数据手册）。

            query: 中文或英文问题，含具体符号（如 HAL_GPIO_Init、TIM1_CH1、PWM 频率）时召回最准。
            sources: "auto" 自动路由，或逗号列表 rm,ds,um,hal,examples。
            k: 返回条数。
            """
            return session.search(query, sources=sources, k=k)

        @tool
        def stm32_doc_graph(kind: str, target: str = "") -> str:
            """查询文档知识图谱（GraphRAG）：符号出现在哪些块、跨语料邻域、调用链、外设相关章节。

            kind:
              - "symbol"：某符号在哪些块出现/定义（带来源、页码），target 如 "HAL_GPIO_Init"
              - "calls"：它调用了谁（2 跳），target 如 "HAL_UART_Init"
              - "callers"：谁调用了它（反向）
              - "peripheral"：哪些文档块讲这个外设 + 器件图信息，target 如 "TIM1"
              - "expand"：给一个 Chroma chunk id，看跨语料邻块
              - "chunk"：块详情（来源/小节/符号/相邻块）
              - "stats"：图谱规模；"cypher"：只读 Cypher（target 是语句）
            """
            return session.doc_graph_query(kind, target)

        @tool
        def project_overview() -> str:
            """查看当前 Keil 工程：器件型号、文件分组、宏定义、头文件路径、缺失文件。"""
            return session.overview()

        @tool
        def read_source(path: str) -> str:
            """读取工作区内文件（相对工作区根，如 Core/Src/main.c 或 MDK-ARM/xxx.uvprojx）。"""
            try:
                return session.read(path)
            except ValueError as exc:
                return str(exc)

        @tool
        def write_source(path: str, content: str) -> str:
            """写入工作区内文件（相对路径）。用于补充手写逻辑或修复编译错误。

            整文件覆盖；写入前请先用 read_source 看清现有内容。
            """
            try:
                return session.write(path, content)
            except ValueError as exc:
                return str(exc)

        @tool
        def generate_application(specs_json: str) -> str:
            """按结构化规格生成 main.c / main.h / stm32f1xx_it.c，并同步工程配置。

            specs_json 示例：
            {"device":"STM32F103RB","sysclk":72000000,
             "pwm":[{"pin":"PA8","freq_hz":1000000,"duty":0.5}],
             "gpio":[{"pin":"PC13","mode":"output_pp","initial":"high"}],
             "uart":[{"instance":"USART1","baud":115200,"dma":false,"printf":true}],
             "timers":[{"timer":"TIM2","freq_hz":1000,"toggle_pin":"PA5"}],
             "loop":{"kind":"toggle","pin":"PC13","interval_ms":500,"message":""}}

            字段都可省略；引脚写 PA8 这种形式。生成器会先校验引脚复用与分频再写文件。
            """
            return session.generate(specs_json)

        @tool
        def stm32_graph_query(kind: str, target: str = "") -> str:
            """查询 STM32F103 器件知识图谱（Neo4j），核对引脚复用 / 定时器通道 / DMA / 中断。

            与 stm32_knowledge_search 的分工：那个查文档（手册原文），这个查关系（确定事实）。
            kind:
              - "pin"：某引脚能做什么，target 如 "PA8"（会列出 TIMx_CHy / USARTx_TX）
              - "timer"：定时器的通道→引脚、DMA、中断，target 如 "TIM3"
              - "usart"：串口的 TX/RX 引脚、DMA 通道、中断，target 如 "USART1"
              - "device"：器件资源（Flash/RAM/定时器/串口），target 如 "STM32F103C8"
              - "pwm_pins"：所有可出 PWM 的引脚
              - "stats"：图谱规模
              - "cypher"：只读 Cypher，target 是语句
            """
            return session.graph_query(kind, target)

        @tool
        def build_project(clean: bool = False) -> str:
            """用 Keil UV4 命令行编译当前工程，返回错误/警告/Program Size。

            clean=True 时先删除中间产物做全量编译。
            """
            return session.build_report(clean=clean)

        @tool
        def auto_fix_build() -> str:
            """按最近一次编译错误自动补齐模块/宏/头文件路径，然后重新编译。"""
            return session.auto_fix()

        return [
            stm32_knowledge_search,
            stm32_graph_query,
            stm32_doc_graph,
            project_overview,
            read_source,
            write_source,
            generate_application,
            build_project,
            auto_fix_build,
        ]

    # ------------------------------------------------------------ 图谱
    def graph_query(self, kind: str, target: str = "") -> str:
        """查 Neo4j 器件知识图谱（引脚复用 / 定时器 / 串口 / DMA / 中断）。"""
        try:
            from ..graphs.graph import GraphStore
        except Exception as exc:  # 驱动缺失时不能阻断生成流程
            return f"图谱不可用：{exc}"
        store = self._graph
        if store is None:
            try:
                store = GraphStore()
            except RuntimeError as exc:
                return str(exc)
            self._graph = store
        try:
            if not store.ping():
                return "图谱不可用：连不上 Neo4j（bolt://localhost:7687），可改用 generate_application 的校验结果"
            text = store.describe(kind, target)
            self.actions.append(f"图谱查询：{kind} {target}".strip())
            return text
        except Exception as exc:
            return f"图谱查询失败：{type(exc).__name__}: {exc}"

    def doc_graph_query(self, kind: str, target: str = "") -> str:
        """查文档知识图谱（符号 / 调用链 / 外设 / 邻域扩展）。"""
        try:
            from ..graphs.graph_docs import doc_graph
        except Exception as exc:  # 驱动缺失时不能阻断生成流程
            return f"文档图谱不可用：{exc}"
        store = self._doc_graph
        if store is None:
            store = doc_graph()
            self._doc_graph = store
        if store is None:
            return "文档图谱不可用：连不上 Neo4j 或还没建图（python -m stm32_rag graph-docs --build）"
        try:
            text = store.describe(kind, target)
            self.actions.append(f"文档图谱：{kind} {target}".strip())
            return text
        except Exception as exc:
            return f"文档图谱查询失败：{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------ 检索
    def search(self, query: str, sources: str = "auto", k: int = 6) -> str:
        try:
            from ..core.config import RetrieveConfig
            from ..corpus.index import Stm32Index
            from ..rag.retrieve import HybridRetriever
            from ..rag.router import MultiSourceRetriever
        except Exception as exc:  # pragma: no cover - 依赖缺失时不影响生成
            return f"检索不可用：{exc}"

        cfg = RetrieveConfig(final_k=max(2, min(k, 8)))
        try:
            if sources.strip() == "auto":
                retriever = MultiSourceRetriever(cfg=cfg)
            else:
                names = [item.strip() for item in sources.split(",") if item.strip()]
                if len(names) == 1:
                    from ..core.config import SOURCE_COLLECTIONS

                    retriever = HybridRetriever(index=Stm32Index(collection=SOURCE_COLLECTIONS.get(names[0], names[0])), cfg=cfg)
                else:
                    retriever = MultiSourceRetriever(sources=names, cfg=cfg)
            hits = retriever.search(query)
        except Exception as exc:  # 索引未建 / 网络异常 / 额度不足
            return f"检索失败：{type(exc).__name__}: {exc}"

        if not hits:
            return "没有检索到相关内容（可能知识库未建立）"
        blocks = []
        for i, hit in enumerate(hits[:k], start=1):
            source = hit.metadata.get("source_key", "")
            symbol = hit.metadata.get("symbol", "")
            header = f"[{i}] {hit.label}" + (f"（{source}）" if source else "") + (f" · {symbol}" if symbol else "")
            snippet = hit.text[:700].replace("\n", " ")
            blocks.append(f"{header}\n{snippet}")
        self.actions.append(f"检索：{query}（{len(blocks)} 条）")
        return "\n\n".join(blocks)


def spec_from_dict(data: dict, session: WorkspaceSession) -> AppSpec:
    """把模型给的 JSON 规格转成 AppSpec。"""
    device = resolve_device(data.get("device") or session.device)
    sysclk = int(data.get("sysclk") or session.sysclk)
    hse = int(data.get("hse") or session.hse)
    spec = AppSpec(device=device, sysclk=sysclk, hse=hse, title=str(data.get("title") or "app"))
    for item in data.get("pwm") or []:
        pin = parse_pin(str(item.get("pin", "")))
        if pin is None:
            raise ValueError(f"PWM 引脚无法解析：{item.get('pin')}")
        spec.pwm.append(
            PwmSpec(
                pin=pin,
                timer=str(item.get("timer") or ""),
                channel=str(item.get("channel") or ""),
                freq_hz=float(item.get("freq_hz") or 1000),
                duty=float(item.get("duty") if item.get("duty") is not None else 0.5),
            )
        )
    for item in data.get("gpio") or []:
        pin = parse_pin(str(item.get("pin", "")))
        if pin is None:
            raise ValueError(f"GPIO 引脚无法解析：{item.get('pin')}")
        spec.gpios.append(
            GpioSpec(
                pin=pin,
                mode=str(item.get("mode") or "output_pp"),
                initial=str(item.get("initial") or ""),
                label=str(item.get("label") or ""),
            )
        )
    for item in data.get("uart") or []:
        instance = str(item.get("instance") or "USART1").upper()
        if instance not in USART_PINS:
            raise ValueError(f"不支持的串口：{instance}")
        tx = parse_pin(str(item.get("tx") or "")) or USART_PINS[instance]["TX"][0]
        rx = parse_pin(str(item.get("rx") or "")) or USART_PINS[instance]["RX"][0]
        spec.uarts.append(
            UartSpec(
                instance=instance,
                baud=int(item.get("baud") or 115200),
                tx=tx,
                rx=rx,
                dma=bool(item.get("dma")),
                printf=bool(item.get("printf", True)),
            )
        )
    for item in data.get("timers") or []:
        pin = parse_pin(str(item.get("toggle_pin") or ""))
        spec.timers.append(
            TimerSpec(
                timer=str(item.get("timer") or "TIM2").upper(),
                freq_hz=float(item.get("freq_hz") or 1000),
                toggle_pin=pin,
            )
        )
    loop = data.get("loop") or None
    if loop:
        pin = parse_pin(str(loop.get("pin") or ""))
        spec.loop = LoopSpec(
            kind=str(loop.get("kind") or "none"),
            pin=pin,
            interval_ms=int(loop.get("interval_ms") or 500),
            message=str(loop.get("message") or ""),
        )
    return spec


def default_workspace(requirement: str) -> Path:
    from ..build.pipeline import _slugify

    return WORKSPACE_ROOT / _slugify(requirement)
