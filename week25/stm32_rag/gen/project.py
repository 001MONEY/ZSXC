"""Keil MDK 工程（.uvprojx）的读取、分析与小步修改。

这是架构图里「工程分析工具」的底座：Agent 不靠猜，直接读工程里的
器件型号、文件组、宏定义、头文件路径，再决定往哪儿加文件、加宏。

两条实测约束（写错会静默失败，排查很久）：

1. 保存必须是 **UTF-8 无 BOM**。带 BOM 时 `UV4 -b` 直接返回 15，
   既不编译也不写日志，看起来像“什么都没发生”。
2. `<Define>` 用**逗号**分隔（`USE_HAL_DRIVER,STM32F103xB`），
   `<IncludePath>` 用**分号**分隔；两者互不通用。
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

# Keil 的 FileType：1=C，2=汇编，4=库，5=头文件（头文件不参与编译，只用于浏览）
FILE_TYPES = {
    ".c": 1,
    ".cpp": 1,
    ".cc": 1,
    ".s": 2,
    ".asm": 2,
    ".a": 4,
    ".lib": 4,
    ".ar": 4,
    ".h": 5,
}

DEFAULT_GROUP = "Application/User"


def file_type_of(name: str) -> int:
    return FILE_TYPES.get(Path(name).suffix.lower(), 1)


@dataclass
class ProjectFile:
    name: str
    path: str  # 相对工程文件（MDK-ARM/*.uvprojx）的路径
    file_type: int
    group: str

    def as_dict(self) -> dict:
        return {"group": self.group, "name": self.name, "path": self.path, "type": self.file_type}


class KeilProject:
    """一个 .uvprojx 的可读写视图（默认只操作第一个 target）。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"找不到 Keil 工程：{self.path}")
        self.tree = ET.parse(self.path)
        self.root = self.tree.getroot()
        self._target = self.root.find("./Targets/Target")
        if self._target is None:
            raise ValueError(f"{self.path.name} 里没有 <Target>，不是有效的 uVision 工程")

    # ------------------------------------------------------------ 基本信息
    @property
    def dir(self) -> Path:
        return self.path.parent

    @property
    def target_name(self) -> str:
        return self._text("./TargetName", "Target 1")

    def _common(self) -> ET.Element:
        node = self._target.find("./TargetOption/TargetCommonOption")
        if node is None:
            raise ValueError("工程缺少 <TargetCommonOption>")
        return node

    def _controls(self) -> ET.Element:
        node = self._target.find("./TargetOption/TargetArmAds/Cads/VariousControls")
        if node is None:  # 极少数工程没有这一段，补一个
            cads = self._target.find("./TargetOption/TargetArmAds/Cads")
            if cads is None:
                cads = ET.SubElement(self._target.find("./TargetOption"), "Cads")
            node = ET.SubElement(cads, "VariousControls")
        return node

    def _text(self, path: str, default: str = "") -> str:
        node = self._target.find(path)
        return (node.text or default).strip() if node is not None else default

    def _set(self, path: str, value: str) -> None:
        node = self._target.find(path)
        if node is None:
            raise ValueError(f"工程里缺少节点 {path}")
        node.text = value

    @property
    def device(self) -> str:
        return self._text("./TargetOption/TargetCommonOption/Device")

    @property
    def vendor(self) -> str:
        return self._text("./TargetOption/TargetCommonOption/Vendor")

    @property
    def pack_id(self) -> str:
        return self._text("./TargetOption/TargetCommonOption/PackID")

    @property
    def output_name(self) -> str:
        return self._text("./TargetOption/TargetCommonOption/OutputName")

    @property
    def output_dir(self) -> str:
        return self._text("./TargetOption/TargetCommonOption/OutputDirectory")

    @property
    def defines(self) -> list[str]:
        raw = self._text("./TargetOption/TargetArmAds/Cads/VariousControls/Define")
        return [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]

    @property
    def include_paths(self) -> list[str]:
        raw = self._text("./TargetOption/TargetArmAds/Cads/VariousControls/IncludePath")
        return [item.strip() for item in raw.split(";") if item.strip()]

    # ------------------------------------------------------------ 文件表
    def groups(self) -> list[str]:
        names: list[str] = []
        for group in self._target.findall("./Groups/Group"):
            name = (group.findtext("GroupName") or "").strip()
            if name:
                names.append(name)
        return names

    def files(self) -> list[ProjectFile]:
        result: list[ProjectFile] = []
        for group in self._target.findall("./Groups/Group"):
            group_name = (group.findtext("GroupName") or "").strip()
            for item in group.findall("./Files/File"):
                name = (item.findtext("FileName") or "").strip()
                path = (item.findtext("FilePath") or "").strip()
                raw_type = (item.findtext("FileType") or "").strip()
                try:
                    file_type = int(raw_type)
                except ValueError:
                    file_type = file_type_of(name or path)
                result.append(ProjectFile(name=name, path=path, file_type=file_type, group=group_name))
        return result

    def has_file(self, rel_path: str) -> bool:
        key = self._key(rel_path)
        return any(self._key(item.path) == key for item in self.files())

    def resolve(self, rel_path: str) -> Path:
        """把工程内路径（`..\\Core\\Src\\main.c` 或 `Core/Src/main.c`）解析成绝对路径。"""
        text = rel_path.replace("\\", os.sep).replace("/", os.sep)
        candidate = Path(text)
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.dir / candidate).resolve()

    def _key(self, rel_path: str) -> str:
        return os.path.normcase(str(self.resolve(self._project_relative(rel_path))))

    def _project_relative(self, rel_path: str) -> str:
        """把用户给的路径换成相对 .uvprojx 的写法。

        两种入参都支持：
        - `Core/Src/main.c`：相对工程根（workspace）；
        - `..\\Core\\Src\\main.c`：已经是相对 MDK-ARM 的写法。
        """
        text = rel_path.strip().replace("/", os.sep)
        if not text:
            return ""
        if text.startswith("..") or Path(text).is_absolute():
            return text.replace(os.sep, "\\")
        workspace = self.dir.parent
        target = workspace / text
        return os.path.relpath(target, self.dir).replace(os.sep, "\\")

    def missing_files(self) -> list[ProjectFile]:
        return [item for item in self.files() if not self.resolve(item.path).is_file()]

    # ------------------------------------------------------------ 修改
    def ensure_group(self, name: str = DEFAULT_GROUP) -> ET.Element:
        groups = self._target.find("./Groups")
        if groups is None:
            groups = ET.SubElement(self._target, "Groups")
        for group in groups.findall("./Group"):
            if (group.findtext("GroupName") or "").strip() == name:
                if group.find("./Files") is None:
                    ET.SubElement(group, "Files")
                return group
        group = ET.SubElement(groups, "Group")
        ET.SubElement(group, "GroupName").text = name
        ET.SubElement(group, "Files")
        return group

    def add_file(self, rel_path: str, group: str = DEFAULT_GROUP, file_type: int | None = None) -> bool:
        """把源文件加进工程；已存在返回 False。"""
        project_path = self._project_relative(rel_path)
        if not project_path or self.has_file(project_path):
            return False
        group_node = self.ensure_group(group)
        files_node = group_node.find("./Files")
        assert files_node is not None
        name = Path(project_path.replace("\\", "/")).name
        item = ET.SubElement(files_node, "File")
        ET.SubElement(item, "FileName").text = name
        ET.SubElement(item, "FileType").text = str(file_type if file_type is not None else file_type_of(name))
        ET.SubElement(item, "FilePath").text = project_path
        return True

    def add_group(self, name: str) -> bool:
        if name in self.groups():
            return False
        self.ensure_group(name)
        return True

    def remove_file(self, rel_path: str) -> bool:
        key = self._key(rel_path)
        for group in self._target.findall("./Groups/Group"):
            files_node = group.find("./Files")
            if files_node is None:
                continue
            for item in list(files_node.findall("./File")):
                path = (item.findtext("FilePath") or "").strip()
                if self._key(path) == key:
                    files_node.remove(item)
                    return True
        return False

    def add_define(self, *names: str) -> list[str]:
        current = self.defines
        added: list[str] = []
        for name in names:
            name = name.strip()
            if name and name not in current:
                current.append(name)
                added.append(name)
        if added:
            self._controls().find("./Define").text = ",".join(current)
        return added

    def set_defines(self, values: list[str]) -> list[str]:
        """整体替换宏定义（逗号分隔）。"""
        items = [item.strip() for item in values if item and item.strip()]
        self._controls().find("./Define").text = ",".join(items)
        return items

    def remove_define(self, name: str) -> bool:
        current = self.defines
        if name not in current:
            return False
        current.remove(name)
        self._controls().find("./Define").text = ",".join(current)
        return True

    def add_include_path(self, *paths: str) -> list[str]:
        current = self.include_paths
        keys = {_normalize(path) for path in current}
        added: list[str] = []
        for path in paths:
            value = self._project_relative(path.strip())
            if value and _normalize(value) not in keys:
                current.append(value)
                keys.add(_normalize(value))
                added.append(value)
        if added:
            self._controls().find("./IncludePath").text = ";".join(current)
        return added

    def remove_include_path(self, path: str) -> bool:
        key = _normalize(path)
        current = self.include_paths
        kept = [item for item in current if _normalize(item) != key]
        if len(kept) == len(current):
            return False
        self._controls().find("./IncludePath").text = ";".join(kept)
        return True

    def set_output_name(self, name: str) -> None:
        self._set("./TargetOption/TargetCommonOption/OutputName", name)

    # ------------------------------------------------------------ 保存 / 汇总
    def save(self) -> Path:
        """写回工程文件（UTF-8 无 BOM，否则 UV4 读不了）。"""
        ET.indent(self.tree, space="  ")
        body = ET.tostring(self.root, encoding="unicode", short_empty_elements=True)
        text = '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n' + body + "\n"
        self.path.write_bytes(text.encode("utf-8"))
        return self.path

    def summary(self) -> dict:
        files = self.files()
        startup = next((item.path for item in files if item.path.lower().endswith((".s", ".asm"))), "")
        ioc = next((item for item in self.dir.parent.glob("*.ioc")), None)
        return {
            "project": str(self.path),
            "target": self.target_name,
            "device": self.device,
            "vendor": self.vendor,
            "pack": self.pack_id,
            "output": f"{self.output_dir}{self.output_name}.axf",
            "startup": startup,
            "cubemx_ioc": str(ioc) if ioc else "",
            "defines": self.defines,
            "include_paths": self.include_paths,
            "groups": {
                name: [item.name for item in files if item.group == name] for name in self.groups()
            },
            "file_count": len(files),
            "missing_files": [item.path for item in self.missing_files()],
        }

    def describe(self) -> str:
        data = self.summary()
        lines = [
            f"工程    : {data['project']}",
            f"目标    : {data['target']}    器件: {data['device']}（{data['vendor']}）",
            f"输出    : {data['output']}",
            f"启动文件: {data['startup'] or '（工程里没有 .s）'}",
            f"宏定义  : {', '.join(data['defines']) or '（无）'}",
            f"头文件  : {'; '.join(data['include_paths']) or '（无）'}",
            f"文件数  : {data['file_count']}",
        ]
        if data["cubemx_ioc"]:
            lines.append(f"CubeMX  : {data['cubemx_ioc']}")
        for group, names in data["groups"].items():
            preview = "、".join(names[:8]) + ("…" if len(names) > 8 else "")
            lines.append(f"  · {group:<28} {len(names):3d} 个  {preview}")
        if data["missing_files"]:
            lines.append(f"  ! 有 {len(data['missing_files'])} 个文件在磁盘上不存在")
        return "\n".join(lines)


def find_project(root: str | Path) -> Path | None:
    """在目录里找 .uvprojx（优先 MDK-ARM 子目录、优先非 Keil 备份）。"""
    root = Path(root).expanduser()
    if root.is_file() and root.suffix.lower() == ".uvprojx":
        return root
    if not root.is_dir():
        return None
    candidates = sorted(root.rglob("*.uvprojx"), key=lambda p: (len(p.parts), len(p.name)))
    if not candidates:
        return None
    preferred = [p for p in candidates if "mdk-arm" in str(p).lower()]
    return (preferred or candidates)[0]


def _normalize(path: str) -> str:
    return path.replace("/", "\\").lstrip(".\\").lower()
