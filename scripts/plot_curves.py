"""把 tensorboard 里的标量画成一张四联图，给人看。

为什么不用 tensorboard 网页：判断"训练是否健康"需要同时看到 total / pred /
var / cov 四条线，网页要来回切。一张图 30 秒就能下结论。

    python scripts/plot_curves.py --tag base_jepa
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

PANELS = [
    ("loss/total", "总损失", "健康：平滑下降\n病态：一条直线（学不动）"),
    ("loss/pred", "预测损失 1-cos", "健康：降到 0.2 以下\n病态：卡在 1.0（完全没学到）"),
    ("loss/var", "VICReg 方差项", "健康：> 0 且波动\n病态：恒为 0（维度塌陷）"),
    ("loss/cov", "VICReg 协方差项", "健康：下降但 > 0\n病态：先降后升（维度共线）"),
]


def load(tag: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    path = Path("outputs") / tag / "tb"
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for k, *_ in PANELS:
        if k in acc.Tags()["scalars"]:
            ev = acc.Scalars(k)
            out[k] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base_jepa")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = load(args.tag)
    if not d:
        print(f"{args.tag} 里没有标量")
        return 1

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), dpi=150)
    for ax, (key, title, hint) in zip(axes.ravel(), PANELS):
        if key in d:
            s, v = d[key]
            ax.plot(s, v, lw=1.2, color="#378ADD")
            ax.set_title(title, fontsize=11, loc="left")
            ax.text(0.02, 0.04, hint, transform=ax.transAxes, fontsize=7.5,
                    color="#888780", va="bottom")
        else:
            ax.set_title(f"{title}（无数据）", fontsize=11, loc="left")
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    fig.suptitle(f"{args.tag}", fontsize=12)
    fig.tight_layout()
    out = Path(args.out or f"outputs/figs/{args.tag}_curves.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"已保存 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
