"""评测指标。三个一起报，缺一个就能被作弊。

约定：所有输入都是二值张量 [B, G_H, G_W]（0/1）。

  iou_all      全格 IoU。包含大量平凡负样本，只能当参考。
  recall_new   新增占据召回：只看"上一帧没有、这一帧有"的格子。
               persistence 在这里恒为 0，所以它测的是"能否提前看到新出现的目标"。
  precision_occ 预测占用里有多少是真的。防"全图标 1"刷召回。
"""

from __future__ import annotations

import torch


def _bin(x: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    return (x > thr).float()


def iou_all(pred: torch.Tensor, gt: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    p, g = _bin(pred, thr), _bin(gt, thr)
    inter = (p * g).sum(dim=(1, 2))
    union = ((p + g) > 0).float().sum(dim=(1, 2))
    return inter / (union + eps)


def recall_new(pred: torch.Tensor, gt: torch.Tensor, prev: torch.Tensor,
               thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """N = {格 : prev=0 且 gt=1}；返回 |pred ∩ N| / |N|。无新增时返回 nan。"""
    p, g, q = _bin(pred, thr), _bin(gt, thr), _bin(prev, thr)
    new = (1 - q) * g
    num = (p * new).sum(dim=(1, 2))
    den = new.sum(dim=(1, 2))
    return torch.where(den > 0, num / (den + eps), torch.full_like(den, float("nan")))


def precision_occ(pred: torch.Tensor, gt: torch.Tensor, thr: float = 0.5,
                  eps: float = 1e-6) -> torch.Tensor:
    p, g = _bin(pred, thr), _bin(gt, thr)
    num = (p * g).sum(dim=(1, 2))
    den = p.sum(dim=(1, 2))
    return torch.where(den > 0, num / (den + eps), torch.full_like(den, float("nan")))


def nanmean(x: list[torch.Tensor]) -> float:
    v = torch.cat([t.flatten() for t in x])
    v = v[~torch.isnan(v)]
    return float(v.mean()) if v.numel() else float("nan")


def latent_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a/b: [B, N, D] 或 [B, T, N, D]，按最后一维求余弦后取均值。"""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()
