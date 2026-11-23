"""调用 Keil MDK（UV4.exe）编译工程，并把构建日志解析成结构化诊断。

三个实测坑（都是“看起来什么都没发生”）：

1. **UV4 是 GUI 程序**：`&` / Popen 不等待进程结束就返回，
   必须同步等待并读取进程退出码，否则永远只会读到上一次的日志。
2. **工程文件带 BOM 时 exit 15**，既不编译也不写日志。
   所以 `project.save()` 一律写 UTF-8 无 BOM。
3. **已有 μVision 实例占用同一工程时**，命令行构建会静默失败。
   这里在拿不到日志时会结束残留 UV4 进程并重试一次（可用 STM32_KILL_STALE_UV4=0 关闭）。

退出码（UV4 官方定义）：0 无错误无警告 / 1 有警告 / 2 有错误 / 3 致命错误 /
11 打不开工程 / 12 器件错误 / 13 无法写工程 / 15 读取工程失败 / 20 license 问题。
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import BUILD_TIMEOUT, KILL_STALE_UV4, UV4_PATH

EXIT_CODES = {
    0: "无错误、无警告",
    1: "编译通过，但有警告",
    2: "有错误，未生成可执行文件",
    3: "致命错误",
    11: "无法打开工程文件",
    12: "器件（Device）配置错误",
    13: "无法写入工程文件",
    15: "读取工程文件失败（常见原因：UTF-8 BOM、XML 损坏、工程被 μVision 占用）",
    20: "License 问题",
}

_COMPILER_ISSUE = re.compile(
    r"^(?P<file>.+?)\((?P<line>\d+)\):\s*(?P<sev>fatal error|error|warning|note):\s*(?P<msg>.*)$",
    re.IGNORECASE,
)
_LINKER_ISSUE = re.compile(
    r"^(?P<file>.+?):\s*(?P<sev>fatal error|error|warning):\s*(?P<code>[A-Z]\d+[A-Z]?):\s*(?P<msg>.*)$",
    re.IGNORECASE,
)
_SIZE_RE = re.compile(r"Program Size:\s*(?P<size>.+)")
_TOTAL_RE = re.compile(r"-\s*(?P<errors>\d+)\s*Error\(s\),\s*(?P<warnings>\d+)\s*Warning\(s\)")
_ELAPSED_RE = re.compile(r"Build Time Elapsed:\s*(?P<elapsed>[\d:]+)")


@dataclass
class BuildIssue:
    severity: str  # error / warning
    message: str
    file: str | None = None
    line: int | None = None
    code: str | None = None

    @property
    def where(self) -> str:
        if self.file and self.line:
            return f"{self.file}:{self.line}"
        return self.file or ""

    def text(self) -> str:
        prefix = f"{self.where} " if self.where else ""
        code = f"[{self.code}] " if self.code else ""
        return f"{prefix}{self.severity}: {code}{self.message}".strip()

    def as_dict(self) -> dict:
        return {
            "severity": self.severity,
            "file": self.file,
            "line": self.line,
            "code": self.code,
            "message": self.message,
        }


@dataclass
class BuildResult:
    ok: bool
    exit_code: int
    errors: list[BuildIssue] = field(default_factory=list)
    warnings: list[BuildIssue] = field(default_factory=list)
    program_size: str = ""
    elapsed: str = ""
    log_path: Path | None = None
    log_text: str = ""
    axf: Path | None = None
    seconds: float = 0.0
    note: str = ""

    def summary(self, limit: int = 8) -> str:
        status = "✅ 编译通过" if self.ok else "❌ 编译失败"
        lines = [f"{status}（退出码 {self.exit_code}：{EXIT_CODES.get(self.exit_code, '未知')}）"]
        if self.program_size:
            lines.append(f"Program Size: {self.program_size}")
        if self.errors:
            lines.append(f"错误 {len(self.errors)} 条：")
            lines.extend(f"  - {issue.text()}" for issue in self.errors[:limit])
            if len(self.errors) > limit:
                lines.append(f"  … 其余 {len(self.errors) - limit} 条见日志")
        if self.warnings:
            lines.append(f"警告 {len(self.warnings)} 条（前 {min(len(self.warnings), 3)}）：")
            lines.extend(f"  - {issue.text()}" for issue in self.warnings[:3])
        if self.axf and self.axf.is_file():
            lines.append(f"产物: {self.axf}")
        if self.note:
            lines.append(f"说明: {self.note}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "errors": [issue.as_dict() for issue in self.errors],
            "warnings": [issue.as_dict() for issue in self.warnings],
            "program_size": self.program_size,
            "elapsed": self.elapsed,
            "log": str(self.log_path or ""),
            "axf": str(self.axf or ""),
            "note": self.note,
        }


@dataclass
class QuickFix:
    """一条可以从构建错误直接推出的修复动作。"""

    action: str  # add_define / add_module / add_file / add_include / manual
    value: str
    reason: str


def find_uv4() -> Path | None:
    if UV4_PATH.is_file():
        return UV4_PATH
    for candidate in (
        Path(r"C:\Keil_v5\UV4\UV4.exe"),
        Path(r"C:\Keil\UV4\UV4.exe"),
        Path(r"D:\Keil_v5\UV4\UV4.exe"),
    ):
        if candidate.is_file():
            return candidate
    return None


def parse_log(text: str) -> tuple[list[BuildIssue], list[BuildIssue], str, str]:
    """解析 UV4 日志。

    UV4 会把 include 链压成一行，用 `.....` 串起来：

        main.c(6): warning: In file included from.....stm32f1xx_hal.h(29): warning: ...
        .....stm32_hal_legacy.h(30): error: 'Legacy/stm32_hal_legacy.h' file not found

    真正有用的是最深一层的诊断，所以按 `....` 切开后取最后一个。
    """
    errors: list[BuildIssue] = []
    warnings: list[BuildIssue] = []
    size = ""
    elapsed = ""
    total_errors = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        total = _TOTAL_RE.search(line)
        if total:
            total_errors = int(total.group("errors"))
        issue = _last_issue(line)
        if issue is not None:
            if issue.severity in ("error", "fatal error"):
                errors.append(issue)
            elif issue.severity == "warning":
                warnings.append(issue)
            continue
        match = _SIZE_RE.search(line)
        if match:
            size = re.sub(r"\s+", " ", match.group("size")).strip()
        match = _ELAPSED_RE.search(line)
        if match:
            elapsed = match.group("elapsed")
    # 去重：同一条错误可能被 include 链重复带出来
    errors = list({(item.file, item.line, item.message): item for item in errors}.values())
    warnings = list({(item.file, item.line, item.message): item for item in warnings}.values())
    # 汇总行说有错、但具体行没解析出来：至少给一条可见诊断，不让自动修复静默失败
    if not errors and total_errors:
        errors.append(
            BuildIssue(severity="error", message=f"编译失败（{total_errors} 个错误，日志中未能定位具体行）")
        )
    return errors, warnings, size, elapsed


def _last_issue(line: str) -> BuildIssue | None:
    """一行里可能串了多段诊断，取最后一段（最靠近错误现场）。"""
    found: BuildIssue | None = None
    for chunk in re.split(r"\.{4,}", line):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = _COMPILER_ISSUE.match(chunk)
        if match:
            found = BuildIssue(
                severity=match.group("sev").lower(),
                message=match.group("msg").strip(),
                file=match.group("file").strip(),
                line=int(match.group("line")),
            )
            continue
        match = _LINKER_ISSUE.match(chunk)
        if match:
            found = BuildIssue(
                severity=match.group("sev").lower(),
                message=match.group("msg").strip(),
                file=match.group("file").strip(),
                code=match.group("code"),
            )
    return found


def suggest_fixes(issues: list[BuildIssue]) -> list[QuickFix]:
    """把典型错误映射成可直接执行的修复动作（供自动修复循环使用）。"""
    fixes: list[QuickFix] = []
    seen: set[tuple[str, str]] = set()

    def add(action: str, value: str, reason: str) -> None:
        if (action, value) in seen:
            return
        seen.add((action, value))
        fixes.append(QuickFix(action, value, reason))

    module_by_symbol = (
        ("HAL_TIM", "tim"),
        ("HAL_TIMEx", "tim"),
        ("HAL_UART", "uart"),
        ("HAL_USART", "usart"),
        ("HAL_ADC", "adc"),
        ("HAL_I2C", "i2c"),
        ("HAL_SPI", "spi"),
        ("HAL_IWDG", "iwdg"),
        ("HAL_WWDG", "wwdg"),
        ("HAL_DMA", "dma"),
        ("HAL_EXTI", "exti"),
        ("HAL_PWR", "pwr"),
        ("HAL_RTC", "rtc"),
    )

    for issue in issues:
        message = issue.message
        lowered = message.lower()

        if "please select first the target stm32f1xx device" in lowered:
            add("add_define", "STM32F103xB", "stm32f1xx.h 需要器件宏来选择寄存器定义")
            continue
        if "stm32f1xx_hal.h" in lowered and ("cannot open" in lowered or "no such file" in lowered):
            add("add_include", r"..\Drivers\STM32F1xx_HAL_Driver\Inc", "缺少 HAL 头文件搜索路径")
            continue
        if "cannot open source input file" in lowered or "no such file or directory" in lowered:
            name = re.search(r"[\"']([^\"']+\.h)[\"']", message)
            if name:
                header = name.group(1)
                if header.startswith("stm32f1xx_hal"):
                    add("add_include", r"..\Drivers\STM32F1xx_HAL_Driver\Inc", f"找不到 {header}")
                elif header.startswith("stm32f1xx") or header.startswith("core_"):
                    add("add_include", r"..\Drivers\CMSIS\Device\ST\STM32F1xx\Include", f"找不到 {header}")
            continue
        if "use of undeclared identifier 'hal_" in lowered or "implicit declaration of function 'hal_" in lowered:
            add("add_define", "USE_HAL_DRIVER", "没有 USE_HAL_DRIVER，HAL 头文件未启用")
            continue

        if "undefined symbol" in lowered or "l6218e" in lowered:
            symbol = message
            match = re.search(r"(?:symbol|Symbol)\s+(\w+)", message)
            if match:
                symbol = match.group(1)
            if symbol == "SystemInit":
                add("add_file", r"..\Core\Src\system_stm32f1xx.c", "缺少 system_stm32f1xx.c（提供 SystemInit）")
                continue
            if symbol in ("Reset_Handler", "__main"):
                add("manual", "startup", "缺少启动文件 startup_stm32f103xb.s")
                continue
            if symbol in ("HAL_IncTick", "HAL_GetTick", "HAL_Delay", "HAL_Init"):
                add("add_module", "hal", f"缺少 HAL 核心（提供 {symbol}）")
                continue
            if symbol.startswith("HAL_TIM"):
                add("add_module", "tim", f"缺少 TIM 模块（提供 {symbol}）")
                continue
            if symbol.startswith("HAL_UART"):
                add("add_module", "uart", f"缺少 UART 模块（提供 {symbol}）")
                continue
            if symbol.startswith("HAL_DMA"):
                add("add_module", "dma", f"缺少 DMA 模块（提供 {symbol}）")
                continue
            if symbol.startswith("HAL_GPIO"):
                add("add_module", "gpio", f"缺少 GPIO 模块（提供 {symbol}）")
                continue
            if symbol.startswith("HAL_RCC"):
                add("add_module", "rcc", f"缺少 RCC 模块（提供 {symbol}）")
                continue
            if symbol.startswith("HAL_NVIC") or symbol.startswith("HAL_SYSTICK"):
                add("add_module", "cortex", f"缺少 Cortex 模块（提供 {symbol}）")
                continue
            for prefix, module in module_by_symbol:
                if symbol.startswith(prefix):
                    add("add_module", module, f"缺少 {module.upper()} 模块（提供 {symbol}）")
                    break
            continue

        if "no space in execution regions" in lowered or "l6406e" in lowered:
            add("manual", "size", "Flash/RAM 装不下：裁剪功能或换更大容量的型号")
        elif "l6220e" in lowered or "l6221e" in lowered:
            add("manual", "scatter", "链接脚本/内存配置与器件不符：检查 IRAM/IROM 设置")
        elif "multiple definition" in lowered or "multiply defined" in lowered:
            add("manual", "duplicate", "同一个函数被定义了两次：检查是否重复生成或重复加入工程文件")
    return fixes


class KeilBuilder:
    """命令行构建封装。"""

    def __init__(self, uv4: str | Path | None = None, timeout: int | None = None, kill_stale: bool | None = None) -> None:
        self.uv4 = Path(uv4) if uv4 else find_uv4()
        self.timeout = timeout or BUILD_TIMEOUT
        self.kill_stale = KILL_STALE_UV4 if kill_stale is None else kill_stale

    # ------------------------------------------------------------ 进程
    @staticmethod
    def running_instances() -> int:
        try:
            output = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq UV4.exe", "/NH"],
                capture_output=True,
                text=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
        except Exception:
            return 0
        return output.lower().count("uv4.exe")

    @staticmethod
    def kill_instances() -> None:
        subprocess.run(
            ["taskkill", "/F", "/IM", "UV4.exe"],
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    # ------------------------------------------------------------ 构建
    def clean(self, project: str | Path) -> int:
        """删掉中间产物强制全量重编（比 UV4 -r 可靠，后者会拉起 GUI）。"""
        out_dir, _ = _outputs(Path(project))
        removed = 0
        for pattern in ("*.o", "*.d", "*.crf", "*.dep", "*.lnp", "*.axf", "*.hex", "*.map"):
            for item in out_dir.glob(pattern):
                try:
                    item.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    def build(self, project: str | Path, clean: bool = False, log_path: str | Path | None = None) -> BuildResult:
        project = Path(project).expanduser().resolve()
        if self.uv4 is None or not Path(self.uv4).is_file():
            return BuildResult(ok=False, exit_code=-1, note=f"找不到 UV4.exe（当前配置 {UV4_PATH}，可用 STM32_UV4 覆盖）")
        if not project.is_file():
            return BuildResult(ok=False, exit_code=-1, note=f"找不到工程文件：{project}")

        if clean:
            self.clean(project)

        log = Path(log_path) if log_path else project.parent / "build.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        if log.exists():
            try:
                log.unlink()
            except OSError:
                pass

        started = time.time()
        result = self._run(project, log)
        note = ""

        # UV4 静默失败：没写日志 + 有残留实例，多半是工程被 μVision 占用
        if not log.exists() and self.kill_stale and self.running_instances():
            self.kill_instances()
            time.sleep(1.0)
            result = self._run(project, log)
            note = "检测到残留的 μVision 进程，已结束并重试一次"

        seconds = time.time() - started
        if not log.exists():
            return BuildResult(
                ok=False,
                exit_code=result.returncode,
                note=note or "UV4 没有生成日志（常见原因：工程文件 BOM/格式错误，或 μVision 正占用工程）",
                seconds=seconds,
            )

        text = log.read_text(encoding="utf-8", errors="replace")
        errors, warnings, size, elapsed = parse_log(text)
        _, axf = _outputs(project)
        # 退出码 1 只表示“有警告”，仍然成功；0/1 且无解析到 error 且有 axf 即算通过。
        # （不做 mtime 新鲜度判断：无改动的增量构建不会重写 axf）
        ok = result.returncode in (0, 1) and not errors and axf.is_file()
        return BuildResult(
            ok=ok,
            exit_code=result.returncode,
            errors=errors,
            warnings=warnings,
            program_size=size,
            elapsed=elapsed,
            log_path=log,
            log_text=text,
            axf=axf if axf.is_file() else None,
            seconds=round(seconds, 2),
            note=note,
        )

    def _run(self, project: Path, log: Path) -> subprocess.CompletedProcess:
        assert self.uv4 is not None
        command = [str(self.uv4), "-b", str(project), "-j0", "-o", str(log)]
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(project.parent),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(command, returncode=-2, stdout="", stderr="timeout")


def _outputs(project: Path) -> tuple[Path, Path]:
    """返回 (产物目录, axf 路径)。不同工程模板的目录不同：

    - 本包生成的工程：`Objects/`
    - CubeMX 生成的工程：`<OutputName>/`（例如 `F103C8T6/F103C8T6.axf`）
    """
    out_dir = project.parent / "Objects"
    name = project.stem
    try:
        from ..gen.project import KeilProject

        obj = KeilProject(project)
        name = obj.output_name or name
        raw = (obj.output_dir or "").strip()
        if raw:
            candidate = Path(raw.replace("\\", "/"))
            if not candidate.is_absolute():
                candidate = project.parent / candidate
            out_dir = candidate.resolve()
    except Exception:
        pass
    return out_dir, out_dir / f"{name}.axf"


def build_project(project: str | Path, clean: bool = False) -> BuildResult:
    """一行式构建入口，供 CLI / Agent 工具调用。"""
    return KeilBuilder().build(project, clean=clean)
