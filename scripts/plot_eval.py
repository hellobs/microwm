"""把评测 JSON 画成主结果图：左 IoU 对比，右 新目标召回与精度。

    python scripts/plot_eval.py --ours outputs/probe_final_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for _n in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if any(_n.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_n]
        plt.rcParams["axes.unicode_minus"] = False
        break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="outputs/probe_final_eval.json")
    ap.add_argument("--out", default="outputs/figs/main_result.png")
    args = ap.parse_args()

    d = json.loads(Path(args.ours).read_text(encoding="utf-8"))
    rows = d["rows"]
    h = [r["h"] for r in rows]
    iou = [r["iou"]["mean"] for r in rows]
    iou_lo = [r["iou"]["ci"][0] for r in rows]
    iou_hi = [r["iou"]["ci"][1] for r in rows]
    base = [r["iou_persist"] for r in rows]
    rnew = [r["rnew"]["mean"] for r in rows]
    rlo = [r["rnew"]["ci"][0] for r in rows]
    rhi = [r["rnew"]["ci"][1] for r in rows]
    pocc = [r["pocc"]["mean"] for r in rows]
    cos = [r["cos_closed"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=150)

    ax = axes[0]
    ax.plot(h, base, "o--", color="#888780", lw=1.4, label="persistence（未来=现在）")
    ax.errorbar(h, iou, yerr=[np.subtract(iou, iou_lo), np.subtract(iou_hi, iou)], fmt="-o",
                color="#378ADD", lw=1.6, capsize=3, label="MicroJEPA（闭环）")
    ax.set_xlabel("预测时域 h（步，每步 200 ms）")
    ax.set_ylabel("全格 IoU")
    ax.set_title("(a) 全格 IoU：模型明显落后", fontsize=11, loc="left")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.errorbar(h, rnew, yerr=[np.subtract(rnew, rlo), np.subtract(rhi, rnew)], fmt="-o",
                color="#1D9E75", lw=1.6, capsize=3, label="$R_{new}$ 新增占据召回")
    ax.plot(h, pocc, "-s", color="#EF9F27", lw=1.4, label="$P_{occ}$ 占用精度")
    ax.axhline(0, color="#A32D2D", lw=1.0, ls=":", label="persistence 的 $R_{new}$（恒为 0）")
    ax.set_xlabel("预测时域 h（步）")
    ax.set_ylim(-0.05, 0.8)
    ax.set_title("(b) 新目标召回：模型有信号，但精度低", fontsize=11, loc="left")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    fig.suptitle(f"MicroJEPA vs persistence（{Path(args.ours).stem}）", fontsize=12)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"已保存 {out}；闭环余弦 " + " / ".join(f"{c:.2f}" for c in cos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
