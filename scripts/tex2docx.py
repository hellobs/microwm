"""把 main.tex 转成可提交的 .docx。

为什么要脚本而不是直接 pandoc，四个坑：
  1. 手动排版的标题区会被 pandoc 再输出一次，造成标题重复；
  2. 作者与单位会被粘成一行；
  3. pandoc 不认识 \\cite，引用编号会整段消失；
  4. pandoc 把 thebibliography 折成一段，并以 "10"（宽度参数）开头。
  3、4 的解法是先读 main.aux 的 \\bibcite 编号，把 \\cite 就地替换成 [2,3] 这样的纯文本；
  再把 main.bbl 里的条目拼成独立段落，pandoc 只负责排版。

    python scripts/tex2docx.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parents[1] / "paper"
SRC = ROOT / "main.tex"
AUX = ROOT / "main.aux"
BBL = ROOT / "main.bbl"
OUT = ROOT / "out" / "submission.docx"


def strip_manual_title(tex: str) -> str:
    """删掉手写的标题区，避免与 pandoc 的 metadata 重复。"""
    start = tex.find("% ---------- 标题区")
    if start == -1:
        return tex
    # 标题区是一个 center 环境，删到 \end{center} 之后
    end = tex.find("\\end{center}", start)
    if end == -1:
        return tex
    end = tex.find("\n", end) + 1
    # 顺手吃掉紧随其后的 \vspace
    tail = tex[end:]
    tail = re.sub(r"^\s*\\vspace\{[^}]*\}\s*\n", "", tail, count=1)
    return tex[:start] + tail


def ensure_metadata(tex: str) -> str:
    """统一 title / author 字段，作者与单位用 \\\\ 分行。"""
    tex = re.sub(r"\\title\{.*?\}\s*\n", "", tex, flags=re.S)
    tex = re.sub(r"\\author\{.*?\}\s*\n(\\date\{\}\s*\n)?", "", tex, flags=re.S)
    meta = (
        "\\title{面向小型无人机场景的动作条件隐空间预测与局部路径修正}\n"
        "\\author{刘子瑄　马婧怡　盛祺　张子睿\\\\[4pt] 华南理工大学}\n"
        "\\date{}\n"
    )
    i = tex.index("\\begin{document}")
    return tex[:i] + meta + tex[i:]


def read_bib_numbers(aux: Path) -> dict[str, int]:
    """从 main.aux 读 key -> 编号，与 PDF 中的编号完全一致。"""
    if not aux.exists():
        raise SystemExit("找不到 main.aux，先跑一次 xelatex + bibtex")
    text = aux.read_text(encoding="utf-8", errors="ignore")
    return {k: int(n) for k, n in re.findall(r"\\bibcite\{([^}]*)\}\{+(\d+)", text)}


def format_numbers(nums: list[int]) -> str:
    """把编号排成 [1-2] / [3, 6] 这类形式，与 natbib 的 compress 一致。"""
    nums = sorted(set(nums))
    parts: list[str] = []
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 1:
            parts.append(f"{nums[i]}-{nums[j]}")
        else:
            parts.append(str(nums[i]))
        i = j + 1
    return "[" + ", ".join(parts) + "]"


def resolve_citations(tex: str, nums: dict[str, int]) -> str:
    """把 \\cite{a,b} 就地换成 [2,3]。pandoc 不认识 \\cite，只能自己来。"""

    def repl(m: re.Match[str]) -> str:
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        got = [nums[k] for k in keys if k in nums]
        if not got:
            print(f"  [警告] 引文无法解析：{m.group(0)}")
            return m.group(0)
        return format_numbers(got)

    return re.sub(r"\\cite\{([^}]*)\}", repl, tex)


def clean_bbl_text(s: str) -> str:
    """把 .bbl 条目里的 LaTeX 命令清掉，只留可读文本。"""
    s = " ".join(s.split())
    s = s.replace("\\newblock", " ")
    while "{\\em " in s or "{\\it " in s:
        s = re.sub(r"\{\\(?:em|it)\s+([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace("~", " ")
    return re.sub(r"\s+", " ", s).strip()


def replace_bibliography(tex: str, bbl: Path) -> str:
    """用 main.bbl 拼出独立的文献段落，替掉 \\bibliography。"""
    if not bbl.exists():
        raise SystemExit("找不到 main.bbl，先跑一次 bibtex")
    raw = bbl.read_text(encoding="utf-8", errors="ignore")
    body = raw.split("\\begin{thebibliography}")[-1]
    items = re.findall(
        r"\\bibitem(?:\[[^\]]*\])?\{[^}]*\}(.*?)(?=\\bibitem(?:\[|\{)|\\end\{thebibliography\})",
        body,
        re.S,
    )
    if not items:
        raise SystemExit("main.bbl 里没解析出条目")

    lines = ["\\section*{参考文献}", ""]
    for i, item in enumerate(items, 1):
        lines.append(f"[{i}] {clean_bbl_text(item)}")
        lines.append("")
    block = "\n".join(lines)

    return re.sub(
        r"\\bibliographystyle\{[^}]*\}\s*\n\s*\\bibliography\{[^}]*\}",
        lambda _m: block,
        tex,
    )


def style_docx(path: Path) -> None:
    """统一中文与西文字体、字号、行距，使 Word 打开即符合课程要求。"""
    try:
        from docx import Document
        from docx.oxml.ns import qn
        from docx.shared import Pt
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
    tex = resolve_citations(tex, read_bib_numbers(AUX))
    tex = replace_bibliography(tex, BBL)

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
