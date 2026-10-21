"""教学图：一个模型"能看见多小"，由 分辨率 ÷ patch 决定。

同一帧、同一批目标，只改 patch 边长，看 token 网格怎么变。
这是理解 ViT 的第一课，也是我们为什么把 patch 从 12 改成 6 的原因。

    python scripts/teach_patch.py --seq uav0000086_00000_v --frame 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path("D:/zzr/data/visdrone/processed/val")


def pick_frame(bx: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    xywh = bx["xywh"][bx["frame"] == n]
    if len(xywh) == 0:
        raise SystemExit(f"第 {n} 帧没有标注框")
    side = np.sqrt(xywh[:, 2] * xywh[:, 3])
    order = np.argsort(np.abs(side - 0.04))[:6]   # 挑几个"小目标"
    return xywh, xywh[order]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="uav0000086_00000_v")
    ap.add_argument("--frame", type=int, default=120)
    ap.add_argument("--out", default="outputs/figs/teach_patch.png")
    args = ap.parse_args()

    seq_dir = ROOT / args.seq
    imgs = np.load(seq_dir / "images.npy", mmap_mode="r")
    bx = np.load(seq_dir / "boxes.npz")
    img = np.asarray(imgs[args.frame])
    h, w = img.shape[:2]
    _, small = pick_frame(bx, args.frame)

    scales = [12, 6, 3]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), dpi=150)

    for ax, p in zip(axes, scales):
        ax.imshow(img, interpolation="nearest")
        gh, gw = h // p, w // p
        for i in range(1, gw):
            ax.axvline(i * p - 0.5, color="w", lw=0.4, alpha=0.7)
        for j in range(1, gh):
            ax.axhline(j * p - 0.5, color="w", lw=0.4, alpha=0.7)

        for x, y, bw, bh in bx["xywh"][bx["frame"] == args.frame]:
            ax.add_patch(Rectangle((x * w, y * h), bw * w, bh * h,
                                   fill=False, ec="#00E5FF", lw=0.6))
        for x, y, bw, bh in small:
            ax.add_patch(Rectangle((x * w, y * h), bw * w, bh * h,
                                   fill=False, ec="#FF3B30", lw=1.4))

        ax.set_title(f"patch {p}px → {gw}×{gh} = {gw * gh} token", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")

    side_px = np.sqrt(small[:, 2] * small[:, 3]).mean() * w
    print(f"红框目标在模型输入上的平均边长 {side_px:.1f} px")
    for p in scales:
        print(f"  patch {p}: 每个目标约 {side_px / p:.2f} 个格子宽")
    print(f"已保存 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
