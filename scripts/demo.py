"""落地演示：一段无人机视频 → 带"预测占用"叠加的 mp4。

画面构成
    主画面   当前帧 + 模型预测的 t+h 时刻占用热力图（红色渐变）
    右上角   延迟核对：t+h 时刻的真实帧 + 真值框（青色轮廓）
    底部     时间码 / 预测时域 / 峰值置信度

为什么右上角要放"真实未来"：这是唯一的自证方式。观众看到红块亮起，
几帧后那个位置真的出现了目标，才会相信预测不是随机的。

用法
    python scripts/demo.py --seq D:/zzr/data/visdrone/processed/val/<seq> \
        --ckpt outputs/ckpt/best.pt --out outputs/demo.mp4

模型契约（demo 只依赖这一条）
    model.rollout(past: [1,T,3,H,W]) -> [1,horizon,1,GH,GW]  概率，非 logits
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

FRAME_W, FRAME_H = 960, 540      # 输出画面尺寸（16:9）


def load_frames(seq_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """返回 (images [N,H,W,3] uint8, occ [N,GH,GW] float32)。"""
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


def overlay_heat(frame: np.ndarray, occ: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """把 GH×GW 的占用概率拉成伪彩色，叠到画面上。"""
    h, w = frame.shape[:2]
    heat = cv2.resize(occ.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    heat = np.clip(heat, 0, 1)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = (heat > 0.25).astype(np.float32)[..., None]
    return (frame * (1 - alpha * mask) + color * (alpha * mask)).astype(np.uint8)


def overlay_gt(frame: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """真值占用画成青色轮廓，表明"这里确实有东西"。"""
    h, w = frame.shape[:2]
    up = cv2.resize(occ.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours((up > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (255, 220, 60), 1)
    return frame


def put_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5,
             color: tuple[int, int, int] = (240, 240, 240)) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="处理后的序列目录（含 images.npy / boxes.npz）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/demo.mp4")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0, help="0 表示整段")
    args = ap.parse_args()

    from microwm.models.microwm import MicroJEPA  # 延迟导入，缺模型时能给出清晰报错

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    model = MicroJEPA(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().cuda()

    imgs, occ = load_frames(Path(args.seq))
    n = imgs.shape[0] if args.max_frames <= 0 else min(imgs.shape[0], args.start + args.max_frames)
    t_in = 8
    if n <= t_in + args.horizon:
        raise SystemExit(f"帧数不足：{n} 帧，至少需要 {t_in + args.horizon + 1}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (FRAME_W, FRAME_H))
    if not writer.isOpened():
        raise SystemExit("VideoWriter 打开失败，检查输出路径与编码器")

    mean = np.array([0.5, 0.5, 0.5], np.float32)
    std = np.array([0.5, 0.5, 0.5], np.float32)

    with torch.no_grad():
        for t in range(t_in - 1, n - args.horizon):
            t0 = t - (t_in - 1)
            clip = imgs[t0:t + 1].astype(np.float32) / 255.0
            clip = (clip - mean) / std
            past = torch.from_numpy(clip).permute(0, 3, 1, 2)[None].cuda()

            pred = model.rollout(past, horizon=args.horizon)[0, -1, 0].float().cpu().numpy()
            peak = float(pred.max())

            frame = cv2.resize(imgs[t], (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            canvas = overlay_heat(frame, pred, args.alpha)

            # 右上角小窗：t+h 的真实帧 + 真值轮廓
            truth = cv2.resize(imgs[t + args.horizon], (288, 162), interpolation=cv2.INTER_LINEAR)
            truth = cv2.cvtColor(truth, cv2.COLOR_RGB2BGR)
            truth = overlay_gt(truth, occ[t + args.horizon])
            canvas[8:170, FRAME_W - 296:FRAME_W - 8] = truth
            cv2.rectangle(canvas, (FRAME_W - 297, 7), (FRAME_W - 7, 171), (200, 255, 200), 1)
            put_text(canvas, f"actual @ t+{args.horizon}", (FRAME_W - 290, 184), 0.42,
                     (200, 255, 200))

            put_text(canvas, "predicted occupancy", (14, 26), 0.55)
            put_text(canvas, f"frame {t:05d}  horizon {args.horizon}  peak {peak:.2f}",
                     (14, FRAME_H - 16), 0.5)

            writer.write(canvas)

    writer.release()
    print(f"已输出 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
