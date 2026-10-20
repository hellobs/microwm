"""关①：用 persistence 基线验证数据管线。

最笨的预测器 = "未来和现在一样"（把当前帧的占据图直接当作未来）。
它必须拿到不像 0 的 IoU；如果它接近 0，说明切片或栅格化写错了，
而不是模型不行。这是整条管线最便宜的自检。

用法：
    python scripts/check_baseline.py --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microwm.data.dataset import ClipDataset  # noqa: E402


def iou(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-6) -> float:
    p = (pred > 0.5).float()
    t = (true > 0.5).float()
    inter = (p * t).sum(dim=(1, 2, 3))
    union = ((p + t) > 0).float().sum(dim=(1, 2, 3))
    return float((inter / (union + eps)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/zzr/data/visdrone/processed")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    ds = ClipDataset(args.root, args.split)
    print(f"{args.split}: {len(ds)} 个 clip 候选")

    horizons = [1, 2, 3, 4, 5]
    ious = {h: [] for h in horizons}
    density = []

    for i in range(min(args.limit, len(ds))):
        s = ds[i]
        now = s["occ_p"][-1]                       # [1, G_H, G_W]
        density.append(float(now.mean()))
        for h in horizons:
            ious[h].append(iou(now[None], s["occ_f"][h - 1][None]))

    dens = float(np.mean(density))
    print(f"当前帧平均占据密度 {dens:.3f}（每格被占的概率）")
    for h in horizons:
        print(f"  horizon {h}: IoU {np.mean(ious[h]):.3f}")

    best = float(np.mean(ious[1]))
    verdict = "通过" if best >= 0.30 else "不通过"
    print(f"\n关① {verdict}（阈值 0.30，实得 {best:.3f}）")
    if verdict != "通过":
        print("提示：检查 bbox 归一化、栅格方向（行=y、列=x）与 clip 索引对齐")
    return 0 if verdict == "通过" else 1


if __name__ == "__main__":
    raise SystemExit(main())
