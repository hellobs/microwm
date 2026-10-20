"""VisDrone2019-MOT 预处理：解压后的原始目录 → 每序列一个 npy + 一个 npz。

为什么不训练时直接读 jpg：一张 1920×1080 的 jpg 解码要 8–15 ms，一个 13 帧的
clip 就是 150 ms，GPU 全程等 CPU。预先把整段序列缩到 192 存成 npy，训练时只是
memmap 切片，几乎零成本。

输出结构：
    <out>/<split>/<seq_id>/images.npy   uint8 (T, S, S, 3)
    <out>/<split>/<seq_id>/boxes.npz    frame / cls / xywh（xywh 已归一化到 0–1）
    <out>/<split>/index.json            [{seq, frames, boxes}]

用法：
    python -m microwm.data.preprocess \
        --root D:/zzr/data/visdrone/raw/VisDrone2019-MOT-val --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

CLS_NAMES = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}


def discover(root: Path) -> tuple[Path, Path]:
    """找 sequences / annotations，兼容解压后多套一层目录。"""
    seq_dirs = [p for p in root.rglob("sequences") if p.is_dir()]
    ann_dirs = [p for p in root.rglob("annotations") if p.is_dir()]
    if not seq_dirs or not ann_dirs:
        raise SystemExit(f"在 {root} 下没找到 sequences/annotations，检查解压是否完整")
    return seq_dirs[0], ann_dirs[0]


def read_annotation(path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """{frame_index(1-based): [(cls, x, y, w, h), ...]}，像素坐标。"""
    out: dict[int, list[tuple[int, float, float, float, float]]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 8:
            continue
        frame = int(parts[0])
        x, y, w, h = (float(v) for v in parts[2:6])
        score, cls = int(parts[6]), int(parts[7])
        if cls == 0 or score == 0 or w <= 1 or h <= 1:   # 忽略区 / 无效框
            continue
        out.setdefault(frame, []).append((cls, x, y, w, h))
    return out


def process_sequence(seq_dir: Path, ann_file: Path, size: int) -> dict | None:
    frames = sorted(p for p in seq_dir.iterdir() if p.suffix.lower() == ".jpg")
    if len(frames) < 8:
        return None

    ann = read_annotation(ann_file)
    if not ann:
        return None

    first = cv2.imread(str(frames[0]))
    if first is None:
        return None
    oh, ow = first.shape[:2]

    imgs = np.zeros((len(frames), size, size, 3), dtype=np.uint8)
    for i, fp in enumerate(frames):
        im = cv2.imread(str(fp))
        if im is None:
            continue
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        imgs[i] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    fr: list[int] = []
    cls: list[int] = []
    xywh: list[tuple[float, float, float, float]] = []
    for fidx, boxes in ann.items():
        i = fidx - 1                     # 标注从 1 开始
        if not (0 <= i < len(frames)):
            continue
        for c, x, y, w, h in boxes:
            fr.append(i)
            cls.append(c)
            xywh.append((x / ow, y / oh, w / ow, h / oh))

    if not fr:
        return None

    return {
        "images": imgs,
        "frame": np.asarray(fr, dtype=np.int32),
        "cls": np.asarray(cls, dtype=np.int8),
        "xywh": np.asarray(xywh, dtype=np.float32).reshape(-1, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="解压后的 VisDrone2019-MOT-* 目录")
    ap.add_argument("--split", default="val", choices=["train", "val", "test-dev"])
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--out", default="D:/zzr/data/visdrone/processed")
    args = ap.parse_args()

    root = Path(args.root)
    seq_root, ann_root = discover(root)
    out_root = Path(args.out) / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    index = []
    seqs = sorted(p for p in seq_root.iterdir() if p.is_dir())
    print(f"{args.split}: {len(seqs)} 个序列 @ {seq_root}")

    for seq_dir in tqdm(seqs, desc="处理"):
        rec = process_sequence(seq_dir, ann_root / f"{seq_dir.name}.txt", args.size)
        if rec is None:
            continue
        dst = out_root / seq_dir.name
        dst.mkdir(exist_ok=True)
        np.save(dst / "images.npy", rec["images"])
        np.savez_compressed(dst / "boxes.npz",
                            frame=rec["frame"], cls=rec["cls"], xywh=rec["xywh"])
        index.append({
            "seq": seq_dir.name,
            "frames": int(rec["images"].shape[0]),
            "boxes": int(rec["frame"].shape[0]),
        })

    (out_root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    total_frames = sum(x["frames"] for x in index)
    total_boxes = sum(x["boxes"] for x in index)
    print(f"完成：{len(index)} 序列 / {total_frames} 帧 / {total_boxes} 个框")
    print(f"输出 {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
