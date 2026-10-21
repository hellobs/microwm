"""演示片：给不看指标的人看。

画面逻辑（一句话）：屏幕先在"马上会出事"的地方亮起一片暖色，
一秒后那块地方真的出现了东西。

刻意不做的事：
  · 不画检测框（那是"检测器"的语言，我们不是）
  · 不显示 loss / IoU / 参数（非技术观众看不懂，只会分心）
  · 不剪辑只挑好帧（被追问就崩）

用法：
    python scripts/demo.py --seq <处理后序列> --ckpt <权重> --out outputs/demo.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

W, H = 1280, 720          # 输出 720p


def load_seq(seq_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    imgs = np.load(seq_dir / "images.npy")
    bx = np.load(seq_dir / "boxes.npz")
    gh, gw = 27, 48
    occ = np.zeros((imgs.shape[0], gh, gw), dtype=np.float32)
    for f, xywh in zip(bx["frame"], bx["xywh"]):
        x0, y0 = int(xywh[0] * gw), int(xywh[1] * gh)
        x1 = int(np.ceil((xywh[0] + xywh[2]) * gw))
        y1 = int(np.ceil((xywh[1] + xywh[3]) * gh))
        occ[f, max(y0, 0):min(y1, gh), max(x0, 0):min(x1, gw)] = 1.0
    return imgs, occ


def attention_layer(prob: np.ndarray, w: int, h: int, alpha: float) -> np.ndarray:
    """把 27×48 的概率图变成一片柔和的暖色区域（不是框）。"""
    p = cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    p = cv2.GaussianBlur(p, (0, 0), sigmaX=w / 90)
    p = np.clip((p - 0.45) / 0.45, 0, 1) ** 1.5      # 只保留高置信区域，否则满屏发光

    # 洋红 → 橙 → 黄（BGR 顺序）。避开纯红：红色在暖色地面上不显眼
    c0 = np.array([255, 0, 255], np.float32)
    c1 = np.array([0, 140, 255], np.float32)
    c2 = np.array([60, 220, 255], np.float32)
    lut = np.zeros((256, 3), np.uint8)
    for i in range(256):
        t = i / 255
        c = c0 + (c1 - c0) * (t * 2) if t < 0.5 else c1 + (c2 - c1) * ((t - 0.5) * 2)
        lut[i] = c.astype(np.uint8)
    color = lut[(p * 255).astype(np.uint8)]
    # 正常混合，不用加法：加法会把画面冲白
    return color.astype(np.float32), (p * alpha)


def gt_outline(frame: np.ndarray, occ: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    up = cv2.resize(occ, (w, h), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours((up > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, cs, -1, (120, 255, 210), 1)
    return frame


def label(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.55) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/demo.mp4")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=300, help="输出帧数上限，0 为整段")
    ap.add_argument("--no-inset", action="store_true", help="去掉右上角的真实未来小窗")
    ap.add_argument("--cpu", action="store_true", help="GPU 被训练占用时用 CPU 渲染")
    args = ap.parse_args()

    from microwm.models.microwm import MicroJEPA

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck["cfg"]
    dev = "cpu" if args.cpu else "cuda"
    model = MicroJEPA(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    imgs, occ = load_seq(Path(args.seq))
    n = imgs.shape[0]
    t_in = 8
    lo = args.start + t_in - 1
    hi = n - args.horizon
    if args.frames:
        hi = min(hi, lo + args.frames)
    if hi <= lo:
        raise SystemExit("帧数不足")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    if not vw.isOpened():
        raise SystemExit("VideoWriter 打不开，检查输出路径")

    inset_w, inset_h = 320, 180
    ix, iy = W - inset_w - 20, 20

    with torch.no_grad():
        for t in range(lo, hi):
            clip = imgs[t - t_in + 1:t + 1].astype(np.float32) / 255.0
            x = torch.from_numpy((clip - 0.5) / 0.5).permute(0, 3, 1, 2)[None].to(dev)
            prob = model.rollout(x, horizon=args.horizon)[0, -1, 0].float().cpu().numpy()

            base = cv2.resize(imgs[t], (W, H), interpolation=cv2.INTER_LINEAR)
            base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR).astype(np.float32)
            color, a = attention_layer(prob, W, H, args.alpha)
            a3 = a[..., None]
            canvas = np.clip(base * (1 - a3) + color * a3, 0, 255).astype(np.uint8)

            if not args.no_inset:
                truth = cv2.resize(imgs[t + args.horizon], (inset_w, inset_h))
                truth = cv2.cvtColor(truth, cv2.COLOR_RGB2BGR)
                truth = gt_outline(truth, occ[t + args.horizon])
                truth = np.clip(truth.astype(np.float32) * 0.92, 0, 255).astype(np.uint8)
                canvas[iy:iy + inset_h, ix:ix + inset_w] = truth
                cv2.rectangle(canvas, (ix, iy), (ix + inset_w, iy + inset_h), (150, 150, 150), 1)
                label(canvas, f"actual in {args.horizon * 6 / 30:.1f}s", (ix, iy + inset_h + 20), 0.42)

            label(canvas, "predicted attention", (20, 32), 0.5)
            vw.write(canvas)

    vw.release()
    print(f"已输出 {out_path}（{hi - lo} 帧 / {args.fps}fps）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
