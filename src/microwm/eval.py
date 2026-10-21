"""评测：把模型和 persistence 放在同一套不可作弊的指标上比。

三个指标的分工（缺一个就能被作弊）：
  iou_all      全格 IoU —— 仅供参考，被 90% 静止格子主导
  R_new        新增占据召回 —— persistence 恒为 0，测"能否提前看到新出现的目标"
  P_occ        占用精度 —— 防"全图标 1"刷 R_new

另外报：
  cos_closed   闭环滚动的隐空间余弦（每个 horizon 一个数）
  cos_open     开环（teacher forcing）作对照

结论必须带 bootstrap 置信区间，否则 0.02 的差距没有意义。

    python -m microwm.eval --ckpt outputs/base_s6/jepa.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data.dataset import ClipDataset
from .metrics import iou_all, latent_cosine, precision_occ, recall_new
from .models.microwm import MicroJEPA


def boot_ci(x: np.ndarray, n: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = [rng.choice(x, size=x.size, replace=True).mean() for _ in range(n)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/base_s6/jepa.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--out", default="")
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
    acc = {k: {h: [] for h in range(H)} for k in ("iou", "rnew", "pocc", "cos_c", "cos_o")}
    base = {k: {h: [] for h in range(H)} for k in ("iou", "rnew", "pocc")}

    with torch.no_grad():
        for i in range(min(args.limit, len(ds))):
            s = ds[i]
            past = s["past"][None].cuda()
            fut = s["future"][None].cuda()
            prev = s["occ_p"][-1].cuda()

            pred = model.rollout(past, horizon=H, closed_loop=True)[0]        # [H,1,GH,GW]
            z_c = model.rollout_latents(past, horizon=H)[0]                   # [H,N,D]
            z_t = model.teacher(fut)[0]                                       # [H,N,D]

            for h in range(H):
                g = s["occ_f"][h].cuda()
                acc["iou"][h].append(float(iou_all(pred[h], g)))
                acc["rnew"][h].append(float(recall_new(pred[h], g, prev)))
                acc["pocc"][h].append(float(precision_occ(pred[h], g)))
                acc["cos_c"][h].append(float(latent_cosine(z_c[h], z_t[h])))
                # persistence：未来 = 现在
                acc["cos_o"][h].append(float("nan"))
                base["iou"][h].append(float(iou_all(prev, g)))
                base["rnew"][h].append(float(recall_new(prev, g, prev)))
                base["pocc"][h].append(float(precision_occ(prev, g)))

    rows = []
    print(f"\n{'h':>2} | {'IoU_all':>16} | {'R_new':>16} | {'P_occ':>16} | {'cos闭环':>7}")
    print("-" * 74)
    for h in range(H):
        line = {"h": h + 1}
        for k in ("iou", "rnew", "pocc"):
            v = np.array(acc[k][h])
            b = np.array(base[k][h])
            lo, hi = boot_ci(v)
            line[k] = {"mean": float(np.nanmean(v)), "ci": [lo, hi]}
            line[k + "_persist"] = float(np.nanmean(b))
        line["cos_closed"] = float(np.nanmean(np.array(acc["cos_c"][h])))
        rows.append(line)
        print(f"{h + 1:>2} | {line['iou']['mean']:6.3f} [{line['iou']['ci'][0]:.3f},{line['iou']['ci'][1]:.3f}]"
              f" | {line['rnew']['mean']:6.3f} [{line['rnew']['ci'][0]:.3f},{line['rnew']['ci'][1]:.3f}]"
              f" | {line['pocc']['mean']:6.3f} [{line['pocc']['ci'][0]:.3f},{line['pocc']['ci'][1]:.3f}]"
              f" | {line['cos_closed']:7.3f}")
    print("\npersistence 对照：IoU "
          + " / ".join(f"{rows[h]['iou_persist']:.3f}" for h in range(H))
          + "；R_new 恒为 0")

    out = Path(args.out or f"outputs/{Path(args.ckpt).parent.name}_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ckpt": args.ckpt, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
