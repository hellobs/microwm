"""数据洞察：四张图，构成论文的数据章。

只读原始标注，不依赖预处理结果，几十秒跑完。
    python scripts/make_stats.py --root D:/zzr/data/visdrone/raw/VisDrone2019-MOT-val

四张图回答四个问题：
  1. 类别分布 —— 这个数据集在干什么
  2. 每帧目标数 —— 场景有多密
  3. 目标尺寸分布 —— 为什么小目标难（并标出 patch 的等效边长）
  4. 出现位置热力图 —— 地平线压缩效应
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

CLS_NAMES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor",
}


_CJK_OK = False


def setup_cjk_font() -> None:
    """尽量用一个能显示中文的字体；找不到就退回英文标签。"""
    global _CJK_OK
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"):
        if any(name.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            _CJK_OK = True
            return
    _CJK_OK = False


def L(zh: str, en: str) -> str:
    """有中文字体就用中文，否则用英文，避免出现方块。"""
    return zh if _CJK_OK else en


def parse(root: Path) -> dict:
    ann_dirs = [p for p in root.rglob("annotations") if p.is_dir()]
    if not ann_dirs:
        raise SystemExit(f"{root} 下没有 annotations")
    ann_dir = ann_dirs[0]

    cls_count: Counter[int] = Counter()
    per_frame: list[int] = []
    areas: list[float] = []
    centers_x: list[float] = []
    centers_y: list[float] = []

    for txt in sorted(ann_dir.glob("*.txt")):
        frame_boxes: dict[int, int] = defaultdict(int)
        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.strip().split(",")
            if len(p) < 8:
                continue
            x, y, w, h = (float(v) for v in p[2:6])
            score, cls = int(p[6]), int(p[7])
            if cls == 0 or score == 0 or w <= 1 or h <= 1:
                continue
            cls_count[cls] += 1
            frame_boxes[int(p[0])] += 1
            areas.append(w * h)
            centers_x.append((x + w / 2) / 1344.0)
            centers_y.append((y + h / 2) / 756.0)
        per_frame.extend(frame_boxes.values())

    return {
        "cls": cls_count,
        "per_frame": np.asarray(per_frame),
        "areas": np.asarray(areas),
        "cx": np.asarray(centers_x),
        "cy": np.asarray(centers_y),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--patch", type=int, default=12, help="模型 patch 边长（缩放后像素）")
    ap.add_argument("--out", default="outputs/figs/stats.png")
    args = ap.parse_args()

    setup_cjk_font()
    d = parse(Path(args.root))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), dpi=150)

    # 1 类别分布（以车为主，人也不少）
    ax = axes[0, 0]
    order = sorted(d["cls"], key=lambda k: -d["cls"][k])
    vals = [d["cls"][k] for k in order]
    ax.bar([CLS_NAMES.get(k, str(k)) for k in order], vals, color="#378ADD")
    ax.set_yscale("log")
    ax.set_ylabel(L("框数量（对数）", "boxes (log)"))
    ax.set_title(L("(a) 类别分布", "(a) Class distribution"), fontsize=11)
    ax.tick_params(axis="x", rotation=38, labelsize=8)

    # 2 每帧目标数
    ax = axes[0, 1]
    ax.hist(d["per_frame"], bins=40, color="#1D9E75")
    ax.axvline(np.median(d["per_frame"]), color="#A32D2D", ls="--", lw=1.2)
    ax.text(np.median(d["per_frame"]), ax.get_ylim()[1] * 0.9,
            f" median={np.median(d['per_frame']):.0f}", color="#A32D2D", fontsize=9)
    ax.set_xlabel(L("每帧有效框数", "boxes per frame"))
    ax.set_ylabel(L("帧数", "frames"))
    ax.set_title(L("(b) 场景密度", "(b) Scene density"), fontsize=11)

    # 3 目标尺寸：这是"为什么小目标难"的核心论据
    ax = axes[1, 0]
    side = np.sqrt(d["areas"])
    ax.hist(side, bins=80, range=(0, np.percentile(side, 99)), color="#EF9F27")
    for p, c, lab in ((args.patch, "#A32D2D", f"patch {args.patch}px"),
                      (args.patch * 2, "#5F5E5A", f"{args.patch * 2}px")):
        ax.axvline(p, color=c, ls="--", lw=1.2, label=lab)
    thresh = (side < args.patch).mean() * 100
    ax.set_xlabel(L("目标等效边长（原图像素，√面积）", "object side (px, sqrt area)"))
    ax.set_ylabel(L("框数量", "boxes"))
    ax.set_title(L(f"(c) 尺寸分布：{thresh:.0f}% 小于一个 patch", f"(c) Size: {thresh:.0f}% below one patch"),
                 fontsize=11)
    ax.legend(fontsize=8)

    # 4 出现位置：地平线压缩
    ax = axes[1, 1]
    h = ax.hist2d(d["cx"], d["cy"], bins=(60, 34), range=[[0, 1], [0, 1]],
                  cmap="magma", cmin=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel(L("水平位置（归一化）", "x (normalised)"))
    ax.set_ylabel(L("垂直位置（归一化，0=顶部）", "y (normalised, 0=top)"))
    ax.set_title(L("(d) 目标出现位置", "(d) Spatial distribution"), fontsize=11)
    fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.03)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")

    print(f"框总数 {len(d['areas'])}，中位边长 {np.median(side):.1f} px，"
          f"小于 1 个 patch 的占 {thresh:.0f}%")
    print(f"每帧中位目标数 {np.median(d['per_frame']):.0f}")
    print(f"已保存 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
