"""诊断：预测器是弱，还是滚动把误差放大了？

三条曲线对比（隐空间余弦，越高越好）：
  开环  每步都喂真实未来编码（teacher forcing）——只测预测器本身
  闭环  喂自己的预测——测"滚得远不远"
  恒等  直接拿最后一帧的隐状态当预测——这是"什么都不做"

判读：
  开环 ≈ 闭环 ≈ 0.4  → 预测器容量不足，滚不滚都一样差
  开环 0.8 / 闭环 0.4 → 滚动漂移，问题在训练策略
  三条都低于恒等      → 预测器连"复制上一帧"都没学会，训练有问题

    python scripts/diag_rollout.py --ckpt outputs/base_s6/jepa.pt
"""

from __future__ import annotations

import argparse

import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from microwm.data.dataset import ClipDataset  # noqa: E402
from microwm.metrics import nanmean  # noqa: E402
from microwm.models.microwm import MicroJEPA  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/base_s6/jepa.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck["cfg"]
    d = cfg["data"]
    model = MicroJEPA(cfg).cuda()
    model.load_state_dict(ck["model"])
    model.eval()

    ds = ClipDataset(d["root"], args.split, clip_len=d["clip_len"], pred_len=d["pred_len"],
                     stride=d["stride"], grid_w=d["grid_w"], grid_h=d["grid_h"],
                     img_w=d["img_w"], img_h=d["img_h"])

    H = args.horizon
    open_c = [[] for _ in range(H)]
    closed_c = [[] for _ in range(H)]
    ident_c = [[] for _ in range(H)]

    cos = torch.nn.functional.cosine_similarity

    with torch.no_grad():
        for i in range(min(args.limit, len(ds))):
            s = ds[i]
            past = s["past"][None].cuda()
            fut = s["future"][None].cuda()

            z_past = model.encode(past)          # [1,8,N,D]
            z_fut = model.teacher(fut)           # [1,5,N,D]  ← 真值目标

            # 开环：每步都喂真值
            z_open = z_past
            for k in range(H):
                pred = model.predictor(z_open)
                open_c[k].append(float(cos(pred, z_fut[:, k], dim=-1).mean()))
                z_open = torch.cat([z_open, z_fut[:, k:k + 1]], dim=1)

            # 闭环：喂自己的预测
            z_closed = z_past
            for k in range(H):
                pred = model.predictor(z_closed)
                closed_c[k].append(float(cos(pred, z_fut[:, k], dim=-1).mean()))
                z_closed = torch.cat([z_closed, pred.unsqueeze(1)], dim=1)

            # 恒等：拿最后一帧当预测
            last = z_past[:, -1:]
            for k in range(H):
                ident_c[k].append(float(cos(last, z_fut[:, k], dim=-1).mean()))

    print(f"\n{'h':>2} | {'开环':>7} | {'闭环':>7} | {'恒等':>7}")
    print("-" * 34)
    for k in range(H):
        print(f"{k + 1:>2} | {nanmean([torch.tensor(open_c[k])]):7.3f} | "
              f"{nanmean([torch.tensor(closed_c[k])]):7.3f} | "
              f"{nanmean([torch.tensor(ident_c[k])]):7.3f}")

    o1 = nanmean([torch.tensor(open_c[0])])
    c5 = nanmean([torch.tensor(closed_c[H - 1])])
    i5 = nanmean([torch.tensor(ident_c[H - 1])])
    print()
    if o1 < i5:
        verdict = "预测器弱于恒等 —— 训练本身有问题，先别调参"
    elif o1 - c5 > 0.15:
        verdict = "滚动漂移明显 —— 预测器还行，是闭环把误差放大了"
    else:
        verdict = "预测器容量不足 —— 开环闭环都低，需要加容量或加训练量"
    print(f"判读：{verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
