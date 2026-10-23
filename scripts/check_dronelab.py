"""校验 DroneLab 占据投影：把栅格叠回原图，肉眼确认盒子对得上障碍。

    python scripts/check_dronelab.py --episode D:/zzr/data/dronelab/s1_seed3/<回合目录>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from microwm.data.dronelab import box_to_grid, load_episode  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--frames", default="0,30,60,90")
    ap.add_argument("--out", default="outputs/dronelab_check.png")
    args = ap.parse_args()

    ep = load_episode(Path(args.episode))
    picks = [int(x) for x in args.frames.split(",")]
    tiles = []
    for fi in picks:
        fi = min(fi, len(ep.frames) - 1)
        f = ep.frames[fi]
        img = Image.open(ep.dir / f["rgb"]).convert("RGB").resize((192, 108), Image.BILINEAR)
        g = box_to_grid(f, ep.meta, 27, 48, 192, 108)
        overlay = np.asarray(img).astype(np.float32)
        up = np.repeat(np.repeat(g, 4, axis=0), 4, axis=1)[:108, :192]
        tint = np.zeros_like(overlay)
        tint[..., 0] = 255.0
        alpha = (up > 0.5)[..., None] * 0.35
        overlay = overlay * (1 - alpha) + tint * alpha
        tiles.append(overlay.astype(np.uint8))
        print(f"frame {fi:4d} 占用率 {g.mean():.3f}  盒子数 {len(f.get('obstacles_world') or [])}")

    canvas = np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:4], axis=1)], axis=0) if len(tiles) >= 4 else np.concatenate(tiles, axis=1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).resize((canvas.shape[1] * 3, canvas.shape[0] * 3), Image.NEAREST).save(out)
    print("已写出", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
