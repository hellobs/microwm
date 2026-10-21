"""把 OpenCV 写出的 mp4v 转成 H.264 —— 不转的话浏览器和 Windows 播放器打不开。

OpenCV 的 VideoWriter 只可靠地提供 mp4v（MPEG-4 Part 2），
而浏览器 <video> 标签只认 H.264（avc1）。这一步不能省。

    python scripts/to_h264.py docs/demo.mp4
    python scripts/to_h264.py docs/demo.mp4 --inplace
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="")
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--crf", type=int, default=20)
    args = ap.parse_args()

    try:
        import imageio_ffmpeg
    except ImportError:
        print("缺少 imageio-ffmpeg：uv pip install imageio-ffmpeg", file=sys.stderr)
        return 1

    src = Path(args.src)
    if not src.exists():
        print(f"找不到 {src}", file=sys.stderr)
        return 1
    dst = Path(args.out) if args.out else src.with_name(src.stem + "_h264.mp4")

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", str(args.crf), "-movflags", "+faststart", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-800:], file=sys.stderr)
        return r.returncode
    print(f"已输出 {dst}")

    if args.inplace:
        dst.replace(src)
        print(f"已替换 {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
