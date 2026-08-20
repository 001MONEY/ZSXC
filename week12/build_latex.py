# -*- coding: utf-8 -*-
"""
md → LaTeX(XeLaTeX) → PDF 构建脚本
流程：
  1. 提取报告中的 ```mermaid 代码块，用 mermaid.ink 渲染成 PNG（images/mermaid/）
  2. 生成构建用 _build_report.md（mermaid 换成 LaTeX 图、清理 raw HTML）
  3. 调用 pandoc + xelatex 生成 PDF

用法：
    python build_latex.py
    python build_latex.py --skip-mermaid   # 跳过流程图渲染（已存在则复用）
"""
import argparse
import base64
import pathlib
import re
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
MD = ROOT / "骨龄评估系统课程设计报告.md"
BUILD_MD = ROOT / "_build_report.md"
MMD_DIR = ROOT / "images" / "mermaid"
HEADER = ROOT / "report-header.tex"
OUT_PDF = ROOT / "骨龄评估系统课程设计报告.pdf"

MERMAID_URL = "https://mermaid.ink/img/{b64}?type=png&bgColor=white"
MMD_PAT = re.compile(r"```mermaid\n(.*?)```", re.S)
DIV_PAT = re.compile(r"<div align=\"center\">.*?</div>\n*---\n*", re.S)


def render_mermaid(code: str, out_png: pathlib.Path):
    """把 mermaid 源码通过 mermaid.ink 渲染为 PNG"""
    b64 = base64.urlsafe_b64encode(code.encode("utf-8")).decode("ascii").rstrip("=")
    url = MERMAID_URL.format(b64=b64)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = r.read()
    if len(data) < 1000 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"mermaid 渲染失败（返回内容异常）: {url}")
    out_png.write_bytes(data)


def fig_block(name: str) -> str:
    """生成居中的 LaTeX 图（供 pandoc 原样透传），限制宽高防止超页"""
    return ("\\begin{figure}[htbp]\n"
            "\\centering\n"
            f"\\includegraphics[width=0.9\\textwidth,height=0.68\\textheight,keepaspectratio]{{images/mermaid/{name}.png}}\n"
            "\\end{figure}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mermaid", action="store_true")
    args = ap.parse_args()

    text = MD.read_text(encoding="utf-8")
    blocks = MMD_PAT.findall(text)
    print(f"[Info] 发现 {len(blocks)} 个 mermaid 图")
    MMD_DIR.mkdir(parents=True, exist_ok=True)

    replaced = []

    def repl(m):
        idx = len(replaced)
        name = f"fig{idx + 1}"
        out = MMD_DIR / f"{name}.png"
        if not (args.skip_mermaid and out.exists()):
            print(f"[Mermaid] 渲染 {name} ...")
            render_mermaid(m.group(1).strip(), out)
            print(f"[Mermaid] {name} -> {out.relative_to(ROOT)}")
        else:
            print(f"[Mermaid] 复用已有 {out.relative_to(ROOT)}")
        replaced.append(name)
        return fig_block(name)

    new_text = MMD_PAT.sub(repl, text)
    new_text = DIV_PAT.sub("", new_text)          # 去掉 raw HTML 标题块

    # ---- 页面分隔与标题处理 ----
    # 中文摘要：不换页（与封面大标题同页），标题居中
    new_text = new_text.replace(
        "## 摘要",
        "\\begin{center}{\\large\\bfseries 摘\\quad 要}\\end{center}")
    # 英文摘要：另起一页，标题居中
    new_text = new_text.replace(
        "## Abstract",
        "\\newpage\n\n\\begin{center}{\\large\\bfseries Abstract}\\end{center}")
    # 目录：另起一页
    new_text = new_text.replace("## 目 录", "\\newpage\n\n## 目 录")
    # 正文：另起一页
    new_text = new_text.replace("# 第 1 章 绪论", "\\newpage\n\n# 第 1 章 绪论")

    BUILD_MD.write_text(new_text, encoding="utf-8")
    print(f"[OK] 构建 md: {BUILD_MD.relative_to(ROOT)}")

    cmd = [
        "pandoc", str(BUILD_MD), "-o", str(OUT_PDF),
        "--pdf-engine=xelatex",
        "-H", str(HEADER),
        "--resource-path", str(ROOT),
        "-V", "mainfont=Times New Roman",
        "-V", "CJKmainfont=SimSun",
        "-V", "CJKsansfont=SimHei",
        "-V", "CJKmonofont=SimSun",
        "-V", "monofont=Consolas",
        "-V", "fontsize=12pt",            # 小四
        "-V", "geometry:margin=2.5cm",
        "-V", "colorlinks=true",
        "-V", "linkcolor=blue",
    ]
    print("[Pandoc] 编译中 ...")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("[失败] pandoc/xelatex 编译出错，见上方日志")
        sys.exit(1)
    print(f"[OK] PDF 已生成: {OUT_PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
