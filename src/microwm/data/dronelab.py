"""DroneLab 数据集适配器：把 UE 采集的回合转成动作条件训练的样本。

每个回合目录包含：
    metadata.json     相机 fx/fy/cx/cy、分辨率、坐标约定
    frames.jsonl      每帧状态、区间等效动作、障碍 AABB、目标
    rgb/%06d.png      第一视角画面

一个样本 = 8 帧观测 + 5 帧未来，附带：
    past/future   图像张量 [T,3,H,W]
    act_past      过去每帧实际执行的动作 [T,4]（机体速度 m/s + 偏航 deg/s）
    act_future    未来每帧实际执行的动作 [T,4]  —— 训练时喂给预测器
    occ_future    未来每帧的占据栅格 [T,1,GH,GW]，由障碍 AABB 投影得到

占据栅格是「图像平面」的定义（与 VisDrone 那套指标一致），不是俯视地图：
无人机在世界里怎么动由动作决定，栅格只回答「画面里哪些位置有东西」。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# 动作归一化：水平 2 m/s、垂直 1 m/s、偏航 90 deg/s（见 SimpleDronePawn.h 默认值）
ACT_SCALE = np.array([2.0, 2.0, 1.0, 90.0], dtype=np.float32)


@dataclass
class Episode:
    dir: Path
    meta: dict
    frames: list[dict]


def load_episode(path: Path) -> Episode:
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    frames = [json.loads(line) for line in (path / "frames.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return Episode(dir=path, meta=meta, frames=frames)


def find_episodes(root: Path) -> list[Path]:
    """递归找出所有含 frames.jsonl 的回合目录。"""
    out = []
    for p in sorted(root.rglob("frames.jsonl")):
        out.append(p.parent)
    return out


# ---------------- 占据栅格 ----------------

def _quat_rotate_inv(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """把世界向量转到相机坐标系（UE 约定：X 前、Y 右、Z 上）。"""
    x, y, z, w = q_xyzw
    # 共轭旋转
    q_conj = np.array([-x, -y, -z, w], dtype=np.float64)
    qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    # q * v * q^-1
    def mul(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ])
    return mul(mul(q_conj, qv), np.array([x, y, z, w], dtype=np.float64))[:3]


def _box_surface_points(center: np.ndarray, extent: np.ndarray, step: float = 0.06, cap: int = 40) -> np.ndarray:
    """在盒子六个面上按固定间距采样，返回 [N,3] 世界坐标点。

    直接在八个角上取包围盒不行：贴近相机或跨越近裁剪面的角点会被投影成巨大坐标，
    包围盒随即鼓成整屏宽度。密采样 + 只保留近裁剪面之前的点，才能得到真实剪影。
    """
    pts = []
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        n1 = int(np.clip(np.ceil(2 * extent[others[0]] / step), 2, cap))
        n2 = int(np.clip(np.ceil(2 * extent[others[1]] / step), 2, cap))
        u = np.linspace(-extent[others[0]], extent[others[0]], n1)
        v = np.linspace(-extent[others[1]], extent[others[1]], n2)
        U, V = np.meshgrid(u, v)
        for sign in (-1.0, 1.0):
            block = np.zeros((U.size, 3), dtype=np.float64)
            block[:, axis] = sign * extent[axis]
            block[:, others[0]] = U.ravel()
            block[:, others[1]] = V.ravel()
            pts.append(block + center)
    return np.concatenate(pts, axis=0)


def _dilate(mask: np.ndarray) -> np.ndarray:
    """3×3 膨胀一次：密采样落到栅格上会有针孔，补一圈即可。"""
    m = mask.astype(bool)
    out = m.copy()
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out.astype(np.float32)


def box_to_grid(frame: dict, meta: dict, grid_h: int, grid_w: int, img_w: int, img_h: int, margin: float = 0.0) -> np.ndarray:
    """把该帧所有障碍盒投影到图像平面，落到占据栅格上。返回 [grid_h, grid_w] 的 0/1。"""
    occ = np.zeros((grid_h, grid_w), dtype=np.float32)
    obstacles = frame.get("obstacles_world") or []
    if not obstacles:
        return occ
    cam_p = np.array(frame["camera_position_world_m"], dtype=np.float64)
    cam_q = np.array(frame["camera_quaternion_ue_xyzw"], dtype=np.float64)
    fx = float(meta["fx"])
    fy = float(meta["fy"])
    cx = float(meta["cx"])
    cy = float(meta["cy"])
    near = 0.3   # 近裁剪面，米

    sx = grid_w / img_w
    sy = grid_h / img_h

    for ob in obstacles:
        c = np.array(ob["center_world_m"], dtype=np.float64)
        e = np.array(ob["extent_world_m"], dtype=np.float64) + margin
        pts = _box_surface_points(c, e)
        rel = pts - cam_p
        x, y, z, w = cam_q
        q_conj = np.array([-x, -y, -z, w])
        # 批量四元数旋转：q_conj * v * q
        vq = np.concatenate([rel, np.zeros((rel.shape[0], 1))], axis=1)
        ax, ay, az, aw = q_conj
        # t = q_conj * vq
        tx = aw * vq[:, 0] + ay * vq[:, 2] - az * vq[:, 1]
        ty = aw * vq[:, 1] - ax * vq[:, 2] + az * vq[:, 0]
        tz = aw * vq[:, 2] + ax * vq[:, 1] - ay * vq[:, 0]
        tw = aw * vq[:, 3] - ax * vq[:, 0] - ay * vq[:, 1] - az * vq[:, 2]
        # d = t * q
        dx = tw * x + tx * w + ty * z - tz * y
        dy = tw * y - tx * z + ty * w + tz * x
        dz = tw * z + tx * y - ty * x + tz * w
        depth = dx
        keep = depth > near
        if not np.any(keep):
            continue
        u = cx + fx * (dy[keep] / depth[keep])
        v = cy - fy * (dz[keep] / depth[keep])
        gx = np.floor(u * sx).astype(np.int64)
        gy = np.floor(v * sy).astype(np.int64)
        ok = (gx >= 0) & (gx < grid_w) & (gy >= 0) & (gy < grid_h)
        if not np.any(ok):
            continue
        occ[gy[ok], gx[ok]] = 1.0
    return _dilate(occ)


# ---------------- Dataset ----------------

class DroneLabClips(Dataset):
    """按固定步长切窗。动作可选置零，用于「有动作 / 无动作」对照。"""

    def __init__(
        self,
        root: str | Path,
        clip_len: int = 8,
        pred_len: int = 5,
        stride: int = 1,
        img_w: int = 192,
        img_h: int = 108,
        grid_w: int = 48,
        grid_h: int = 27,
        use_actions: bool = True,
        episode_limit: int | None = None,
        split: str | None = None,
        val_ratio: float = 0.2,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.clip_len = clip_len
        self.pred_len = pred_len
        self.stride = stride
        self.img_w = img_w
        self.img_h = img_h
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.use_actions = use_actions

        episodes = [load_episode(p) for p in find_episodes(self.root)]
        # 按回合切分训练/验证，避免同一段视频同时出现在两边
        if split in ("train", "val"):
            n = len(episodes)
            cut = max(1, int(round(n * (1 - val_ratio))))
            episodes = episodes[:cut] if split == "train" else episodes[cut:]
        if episode_limit:
            episodes = episodes[:episode_limit]
        self.episodes = episodes
        if not self.episodes:
            raise FileNotFoundError(f"{self.root} 下没有找到可用的 frames.jsonl")

        self.index: list[tuple[int, int]] = []
        need = self.clip_len + self.pred_len
        for ei, ep in enumerate(self.episodes):
            n = len(ep.frames)
            if n < need:
                continue
            for start in range(0, n - need + 1, self.stride):
                self.index.append((ei, start))

        # 占据栅格缓存：同一回合只投影一次
        self._occ_cache: dict[int, np.ndarray] = {}
        # 图像解码缓存：PNG 逐帧解码是训练的主要瓶颈（每步要解 50 多张），
        # 首次访问时整回合解成 uint8 数组存盘，之后按 memmap 读。
        self.cache_dir = self.root / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _images_all(self, ei: int) -> np.ndarray:
        """整回合图像 [T,H,W,3] uint8，带磁盘缓存。"""
        ep = self.episodes[ei]
        path = self.cache_dir / f"{ep.dir.name}_{self.img_w}x{self.img_h}.npy"
        if path.exists():
            try:
                arr = np.load(path, mmap_mode="r")
                if arr.shape[0] == len(ep.frames):
                    return arr
            except Exception:
                pass
        arr = np.stack(
            [self._decode(ep, f) for f in ep.frames], axis=0
        ).astype(np.uint8)
        np.save(path, arr)
        return arr

    def _decode(self, ep: Episode, frame: dict) -> np.ndarray:
        rel = frame.get("rgb") or ""
        with Image.open(ep.dir / rel) as im:
            im = im.convert("RGB")
            if im.size != (self.img_w, self.img_h):
                im = im.resize((self.img_w, self.img_h), Image.BILINEAR)
            return np.asarray(im, dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.index)

    def _occ_all(self, ei: int) -> np.ndarray:
        if ei in self._occ_cache:
            return self._occ_cache[ei]
        ep = self.episodes[ei]
        rects = [
            box_to_grid(f, ep.meta, self.grid_h, self.grid_w, self.img_w, self.img_h)
            for f in ep.frames
        ]
        arr = np.stack(rects, axis=0)
        self._occ_cache[ei] = arr
        return arr

    def _image(self, ei: int, fi: int) -> np.ndarray:
        return self._images_all(ei)[fi]

    def __getitem__(self, idx: int) -> dict:
        ei, start = self.index[idx]
        ep = self.episodes[ei]
        need = self.clip_len + self.pred_len
        frames = ep.frames[start : start + need]

        imgs = np.stack([self._image(ei, start + i) for i in range(need)], axis=0)
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div_(255.0)

        acts = np.stack([
            np.array(f["action_velocity_interval_body_mps"] + [f["action_yaw_rate_interval_deg_s"]], dtype=np.float32)
            for f in frames
        ])
        acts = acts / ACT_SCALE
        if not self.use_actions:
            acts = np.zeros_like(acts)

        occ = self._occ_all(ei)[start : start + need]
        occ = torch.from_numpy(occ).unsqueeze(1)   # [T,1,GH,GW]

        return {
            "past": imgs[: self.clip_len],
            "future": imgs[self.clip_len :],
            "act_past": torch.from_numpy(acts[: self.clip_len]),
            "act_future": torch.from_numpy(acts[self.clip_len :]),
            "occ": occ,                                  # 全部 T 帧，探针阶段用
            "occ_p": occ[: self.clip_len],               # 过去帧
            "occ_f": occ[self.clip_len :],               # 未来帧
            "episode": ei,
            "start": start,
        }


def summarise(episodes: Iterable[Path]) -> dict:
    """快速统计：帧数、图像数、结果分布、占用率。"""
    eps = [load_episode(p) for p in episodes]
    frames = sum(len(e.frames) for e in eps)
    results: dict[str, int] = {}
    for e in eps:
        r = e.frames[-1].get("result", "?")
        results[r] = results.get(r, 0) + 1
    occ_rates = []
    for e in eps[:5]:
        for f in e.frames[:20]:
            g = box_to_grid(f, e.meta, 27, 48, 192, 108)
            occ_rates.append(float(g.mean()))
    return {
        "episodes": len(eps),
        "frames": frames,
        "results": results,
        "occ_rate_mean": float(np.mean(occ_rates)) if occ_rates else 0.0,
    }
