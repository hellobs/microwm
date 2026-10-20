"""数据集：从预处理好的 npy 里切 clip，并把 bbox 栅格化成占据图。

一条样本 = 13 帧（8 帧输入 + 5 帧预测）× 每帧一张占据图。
图像走 memmap，内存占用与序列总长无关。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

G_W, G_H = 48, 27  # 占据栅格尺寸：192/4 × 108/4，与 config 一致


def rasterize(xywh: np.ndarray, gw: int = G_W, gh: int = G_H) -> np.ndarray:
    """把归一化 bbox 画进 gh×gw 的二值占据图。

    直接取整数格范围填充，不做面积阈值——宁可略大（recall 优先），
    因为小目标（20–40 px）在 48 格上本来就只占 1 格。
    """
    occ = np.zeros((gh, gw), dtype=np.float32)
    if len(xywh) == 0:
        return occ
    x0 = np.floor(xywh[:, 0] * gw).astype(int)
    y0 = np.floor(xywh[:, 1] * gh).astype(int)
    x1 = np.ceil((xywh[:, 0] + xywh[:, 2]) * gw).astype(int)
    y1 = np.ceil((xywh[:, 1] + xywh[:, 3]) * gh).astype(int)
    np.clip(x0, 0, gw, out=x0)
    np.clip(y0, 0, gh, out=y0)
    np.clip(x1, 0, gw, out=x1)
    np.clip(y1, 0, gh, out=y1)
    for a, b, c, d in zip(y0, y1, x0, x1):
        if b > a and d > c:
            occ[a:b, c:d] = 1.0
    return occ


class ClipDataset(Dataset):
    """返回 dict：
        past   float32 [T_in, 3, S, S]   归一化到 [-1, 1]
        future float32 [T_out, 3, S, S]
        occ_p  float32 [T_in, 1, G, G]
        occ_f  float32 [T_out, 1, G, G]
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        clip_len: int = 8,
        pred_len: int = 5,
        stride: int = 2,
        grid_w: int = G_W,
        grid_h: int = G_H,
    ) -> None:
        self.root = Path(root) / split
        self.clip_len = clip_len
        self.pred_len = pred_len
        self.stride = stride
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.span = (clip_len + pred_len - 1) * stride + 1

        index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        self.items: list[tuple[str, int]] = []
        for rec in index:
            last = rec["frames"] - self.span
            for s in range(0, max(last, -1) + 1, stride):
                self.items.append((rec["seq"], s))
        if not self.items:
            raise RuntimeError(f"{self.root} 里没有可用的 clip（序列太短或 index.json 为空）")

        self._cache: dict[str, tuple[np.ndarray, dict]] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, seq: str) -> tuple[np.ndarray, dict]:
        if seq not in self._cache:
            if len(self._cache) > 8:          # 控制驻留序列数
                self._cache.pop(next(iter(self._cache)))
            imgs = np.load(self.root / seq / "images.npy", mmap_mode="r")
            bx = np.load(self.root / seq / "boxes.npz")
            per_frame: dict[int, np.ndarray] = {}
            for f in np.unique(bx["frame"]):
                per_frame[int(f)] = bx["xywh"][bx["frame"] == f]
            self._cache[seq] = (imgs, per_frame)
        return self._cache[seq]

    def __getitem__(self, i: int) -> dict:
        seq, start = self.items[i]
        imgs, per_frame = self._load(seq)

        total = self.clip_len + self.pred_len
        idx = [start + k * self.stride for k in range(total)]

        frames = np.stack([np.asarray(imgs[j], dtype=np.float32) / 127.5 - 1.0 for j in idx])
        occs = np.stack([rasterize(per_frame.get(j, np.zeros((0, 4), np.float32)),
                                   self.grid_w, self.grid_h)
                         for j in idx])

        t_in = self.clip_len
        return {
            "past": torch.from_numpy(frames[:t_in]).permute(0, 3, 1, 2).contiguous(),
            "future": torch.from_numpy(frames[t_in:]).permute(0, 3, 1, 2).contiguous(),
            "occ_p": torch.from_numpy(occs[:t_in]).unsqueeze(1),
            "occ_f": torch.from_numpy(occs[t_in:]).unsqueeze(1),
            "seq": seq,
        }
