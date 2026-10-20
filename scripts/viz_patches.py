"""把一张图切成 patch，并演示"打乱顺序会怎样"。

这是 W1 的直觉起点：图像 = 一串 token。
右图把同样的 144 块打乱重排——内容一个像素没少，但意义全没了。
这就是为什么 Transformer 必须有位置编码。

用法：
    python scripts/viz_patches.py --image <路径> --patch 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--out", default="outputs/figs/patches.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"读不到图片: {args.image}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)

    p = args.patch
    n = args.size // p                      # 每边几块
    tokens = img.reshape(n, p, n, p, 3).swapaxes(1, 2).reshape(n * n, p, p, 3)

    print(f"图像 {args.size}×{args.size} → patch {p}×{p} → {n}×{n} = {n * n} 个 token")
    print(f"每个 token 展平后是 {p * p * 3} 维向量（RGB 三通道）")
    print(f"整帧序列长度 = {n * n}，8 帧就是 {n * n * 8} 个 token")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n * n)
    shuffled = tokens[order].reshape(n, p, n, p, 3).swapaxes(1, 2).reshape(args.size, args.size, 3)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=150)
    axes[0].imshow(img)
    axes[0].set_title("原图", fontsize=11)
    axes[1].imshow(img)
    for i in range(1, n):
        axes[1].axhline(i * p - 0.5, color="w", lw=0.6, alpha=0.8)
        axes[1].axvline(i * p - 0.5, color="w", lw=0.6, alpha=0.8)
    axes[1].set_title(f"切成 {n}×{n} 块 = {n * n} 个 token", fontsize=11)
    axes[2].imshow(shuffled)
    axes[2].set_title("同样的块，打乱顺序", fontsize=11)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"已保存 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
