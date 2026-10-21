"""把 main.tex 转成可提交的 .docx。

为什么要脚本而不是直接 pandoc：
  1. 手动排版的标题块会被 pandoc 再输出一次，造成标题重复；
  2. 作者与单位会被粘成一行；
  3. Word 默认样式是西文字体，中文显示不对。

做法：去掉手动标题块，改用 pandoc 的 metadata（\\title / \\author），
转换后再用 python-docx 统一字体、字号与行距。

    python scripts/tex2docx.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parents[1] / "paper"
SRC = ROOT / "main.tex"
OUT = ROOT / "out" / "submission.docx"


def strip_manual_title(tex: str) -> str:
    """删掉手写的标题区，避免与 pandoc 的 metadata 重复。"""
    start = tex.find("% ---------- 标题区")
    if start == -1:
        return tex
    # 标题区结束于第一条 \rule 之后的换行
    end = tex.find("\\rule{\\textwidth}{0.6pt}", start)
    if end == -1:
        return tex
    end = tex.find("\n", end) + 1
    return tex[:start] + tex[end:]


def ensure_metadata(tex: str) -> str:
    """统一 title / author 字段，作者与单位用 \\\\ 分行。"""
    tex = re.sub(r"\\title\{.*?\}\s*\n", "", tex, flags=re.S)
    tex = re.sub(r"\\author\{.*?\}\s*\n(\\date\{\}\s*\n)?", "", tex, flags=re.S)
    meta = (
        "\\title{面向小型无人机场景的动作条件隐空间预测与局部路径修正}\n"
        "\\author{张子睿　盛祺　刘子瑄　马婧怡\\\\[4pt] 华南理工大学}\n"
        "\\date{}\n"
    )
    i = tex.index("\\begin{document}")
    return tex[:i] + meta + tex[i:]


def style_docx(path: Path) -> None:
    """统一中文与西文字体、字号、行距，使 Word 打开即符合课程要求。"""
    try:
        from docx import Document
        from docx.shared import Pt
        from docx.oxml.ns import qn
    except ImportError:
        print("  （未安装 python-docx，跳过样式修正）")
        return

    doc = Document(str(path))
    for style_name in ("Normal", "Body Text", "First Paragraph"):
        try:
            st = doc.styles[style_name]
        except KeyError:
            continue
        st.font.name = "Times New Roman"
        st.font.size = Pt(12)
        rpr = st.element.get_or_add_rPr()
        rpr.get_or_add_rFonts().set(qn("w:eastAsia"), "宋体")
        pf = st.paragraph_format
        pf.line_spacing = 1.5
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)

    for name, size in (("Title", 18), ("Heading 1", 15), ("Heading 2", 13)):
        try:
            st = doc.styles[name]
        except KeyError:
            continue
        st.font.size = Pt(size)
        st.font.name = "Times New Roman"
        rpr = st.element.get_or_add_rPr()
        rpr.get_or_add_rFonts().set(qn("w:eastAsia"), "黑体")

    doc.save(str(path))


def main() -> int:
    tex = SRC.read_text(encoding="utf-8")
    tex = ensure_metadata(strip_manual_title(tex))

    tmp = ROOT / "out" / "_tmp.tex"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_text(tex, encoding="utf-8")

    pypandoc.convert_file(
        str(tmp),
        "docx",
        outputfile=str(OUT),
        extra_args=["--resource-path=" + str(ROOT)],
    )
    print(f"已转换 {OUT}")
    style_docx(OUT)
    print(f"尺寸 {OUT.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
