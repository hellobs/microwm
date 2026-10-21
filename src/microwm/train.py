"""训练入口。

两个阶段，分开跑，理由写在下面：

  阶段 1  jepa   训练编码器 + 预测器（EMA 目标 + VICReg）
  阶段 2  probe  冻结编码器，只训练探针（把表征映射到占据图）

为什么分开：如果探针和编码器一起训，探针就成了一个"解码器"，
"在表征空间预测"这个前提就被破坏了。分开之后，探针的分数才真正
反映"冻结表征里有多少可用信息"——这是 JEPA 的标准评测协议。

用法：
    python -m microwm.train --config configs/smoke.yaml --phase jepa
    python -m microwm.train --config configs/smoke.yaml --phase probe
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data.dataset import ClipDataset
from .metrics import iou_all, latent_cosine, nanmean, precision_occ, recall_new
from .models.microwm import MicroJEPA


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_loaders(cfg: dict, batch: int) -> tuple[DataLoader, DataLoader]:
    d = cfg["data"]
    kw = dict(clip_len=d["clip_len"], pred_len=d["pred_len"], stride=d["stride"],
              grid_w=d["grid_w"], grid_h=d["grid_h"],
              img_w=d["img_w"], img_h=d["img_h"])
    train = ClipDataset(d["root"], "train", **kw)
    val = ClipDataset(d["root"], "val", **kw)
    return (
        DataLoader(train, batch_size=batch, shuffle=True, num_workers=0, drop_last=True),
        DataLoader(val, batch_size=batch, shuffle=False, num_workers=0, drop_last=True),
    )


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model: MicroJEPA, loader: DataLoader, max_batches: int = 20,
             horizons: int = 5) -> dict:
    """闭环滚动评测：全程不看真实未来。"""
    model.eval()
    out = {"iou": [[] for _ in range(horizons)],
           "rnew": [[] for _ in range(horizons)],
           "pocc": [[] for _ in range(horizons)],
           "cos_closed": [[] for _ in range(horizons)]}
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        past = b["past"].cuda(non_blocking=True)
        fut = b["future"].cuda(non_blocking=True)
        pred = model.rollout(past, horizon=horizons, closed_loop=True)   # [B,H,1,GH,GW]
        z_true = model.teacher(fut)                                      # [B,H,N,D]
        z_pred = model.rollout_latents(past, horizon=horizons)           # [B,H,N,D]

        prev = b["occ_p"][:, -1].cuda()
        for h in range(horizons):
            p, g = pred[:, h, 0], b["occ_f"][:, h, 0].cuda()
            out["iou"][h].append(iou_all(p, g))
            out["rnew"][h].append(recall_new(p, g, prev))
            out["pocc"][h].append(precision_occ(p, g))
            out["cos_closed"][h].append(latent_cosine(z_pred[:, h], z_true[:, h]))
    model.train()
    return {k: [nanmean(x) for x in v] for k, v in out.items()}


def train_jepa(cfg: dict, args) -> Path:
    t = cfg["train"]
    out_dir = Path("outputs") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")

    train_loader, val_loader = build_loaders(cfg, t["batch"])
    model = MicroJEPA(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    total = args.steps or t["steps_per_epoch"] * t["epochs"]
    warm = int(total * 0.03) if t.get("warmup_epochs") else 0
    accum = t.get("grad_accum", 1)

    step, t0 = 0, time.time()
    pbar = tqdm(total=total, desc="jepa")
    while step < total:
        for batch in train_loader:
            if step >= total:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, t["lr"], warm)

            past = batch["past"].cuda(non_blocking=True)
            fut = batch["future"].cuda(non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                r = model(past, fut)
                loss = r["loss"] / accum
            loss.backward()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("grad_clip", 1.0))
                opt.step()
                opt.zero_grad(set_to_none=True)
                model.update_teacher()

            if step % 10 == 0:
                writer.add_scalar("loss/total", float(r["loss"]), step)
                writer.add_scalar("loss/pred", float(r["pred"]), step)
                writer.add_scalar("loss/var", float(r["var"]), step)
                writer.add_scalar("loss/cov", float(r["cov"]), step)
                pbar.set_postfix(loss=f"{float(r['loss']):.3f}",
                                 pred=f"{float(r['pred']):.3f}",
                                 var=f"{float(r['var']):.3f}")

            if step > 0 and step % args.eval_every == 0:
                m = evaluate(model, val_loader, max_batches=10, horizons=cfg["data"]["pred_len"])
                for h, v in enumerate(m["cos_closed"]):
                    writer.add_scalar(f"val/cos_closed_h{h + 1}", v, step)
                print(f"\n[step {step}] 闭环隐空间余弦: "
                      + "  ".join(f"h{h + 1}={v:.3f}" for h, v in enumerate(m["cos_closed"])))

            step += 1
            pbar.update(1)
        pbar.close()

    ckpt = out_dir / "jepa.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, ckpt)
    writer.close()
    print(f"训练完成 {time.time() - t0:.0f}s → {ckpt}")
    return ckpt


def train_probe(cfg: dict, args, jepa_ckpt: Path) -> Path:
    """冻结编码器，只训练探针。输入是真实帧的编码结果，目标是该帧的占据图。"""
    t = cfg["train"]
    out_dir = Path("outputs") / args.tag
    writer = SummaryWriter(out_dir / "tb_probe")

    train_loader, _ = build_loaders(cfg, t["batch"])
    model = MicroJEPA(cfg).cuda()
    model.load_state_dict(torch.load(jepa_ckpt, map_location="cpu")["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.probe.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(model.probe.parameters(), lr=1e-3)
    total = args.steps or 800
    step, t0 = 0, time.time()
    pos_weight = torch.tensor([20.0], device="cuda")

    pbar = tqdm(total=total, desc="probe")
    while step < total:
        for batch in train_loader:
            if step >= total:
                break
            x = torch.cat([batch["past"], batch["future"]], dim=1).cuda(non_blocking=True)
            y = torch.cat([batch["occ_p"], batch["occ_f"]], dim=1).cuda(non_blocking=True)
            with torch.no_grad():
                z = model.encoder(x)                       # [B,T,N,D]
            logits = model.probe(z.flatten(0, 1), model.tok_h, model.tok_w)  # [B*T,1,GH,GW]
            loss = F.binary_cross_entropy_with_logits(
                logits, y.flatten(0, 1), pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 10 == 0:
                writer.add_scalar("probe/loss", float(loss), step)
                pbar.set_postfix(loss=f"{float(loss):.4f}")
            step += 1
            pbar.update(1)
        pbar.close()

    ckpt = out_dir / "model.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, ckpt)
    writer.close()
    print(f"探针训练完成 {time.time() - t0:.0f}s → {ckpt}")
    return ckpt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--phase", default="jepa", choices=["jepa", "probe"])
    ap.add_argument("--steps", type=int, default=0, help="0 = 用配置里的值")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    torch.manual_seed(cfg.get("seed", 42))
    args.tag = args.tag or f"{Path(args.config).stem}_{args.phase}"

    if args.phase == "jepa":
        ckpt = train_jepa(cfg, args)
        print(json.dumps({"ckpt": str(ckpt)}, ensure_ascii=False))
    else:
        jepa = Path("outputs") / args.tag.replace("_probe", "_jepa") / "jepa.pt"
        train_probe(cfg, args, jepa)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
