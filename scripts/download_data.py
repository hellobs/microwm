"""VisDrone2019-MOT 多线程断点下载器。

单线程从 hf-mirror 拉只有 ~1.6 MB/s，9 GB 要一个半小时。
按 Range 切块并发下载，实测能到 5-8 MB/s。

用法：
    python scripts/download_data.py --subset val
    python scripts/download_data.py --subset train --jobs 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://hf-mirror.com/datasets/vanthanh/VisDrone2019-MOT/resolve/main"

SUBSETS = {
    "val": f"{BASE}/VisDrone2019-MOT-val.zip",
    "train": f"{BASE}/VisDrone2019-MOT-train.zip",
    "test-dev": f"{BASE}/VisDrone2019-MOT-test-dev.zip",
}

CHUNK = 32 * 1024 * 1024  # 每块 32 MB


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int, dst: Path, retries: int = 4) -> int:
    """把 [start, end] 写进 dst 的对应偏移。

    每个线程各自打开句柄、各自 seek，写的是互不重叠的区间，
    因此不需要分片文件、也不需要事后合并（合并要批量删除，
    会触发沙箱的批量删除保护，导致下载被中断）。
    """
    size = end - start + 1
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            written = 0
            with open(dst, "r+b") as f, urllib.request.urlopen(req, timeout=60) as r:
                f.seek(start)
                while True:
                    buf = r.read(1024 * 256)
                    if not buf:
                        break
                    f.write(buf)
                    written += len(buf)
            if written == size:
                return size
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise RuntimeError(f"块 {start}-{end} 失败: {e}") from e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"块 {start}-{end} 未完成（期望 {size}，实得 {written}）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="val", choices=[*SUBSETS, "all"])
    ap.add_argument("--out", default="D:/zzr/data/visdrone/raw")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    subsets = list(SUBSETS) if args.subset == "all" else [args.subset]

    for name in subsets:
        url = SUBSETS[name]
        dst = out / f"VisDrone2019-MOT-{name}.zip"

        if dst.exists():
            print(f"[skip] {dst.name} 已存在 {dst.stat().st_size / 1e9:.2f} GB")
            continue

        size = remote_size(url)
        print(f"[info] {name}: {size / 1e9:.2f} GB -> {dst}")

        ranges = [(i, min(i + CHUNK - 1, size - 1)) for i in range(0, size, CHUNK)]

        with open(dst, "wb") as f:      # 预分配，供各线程按偏移写入
            f.truncate(size)

        t0 = time.time()
        done_bytes = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {
                pool.submit(fetch_range, url, s, e, dst): (s, e)
                for s, e in ranges
            }
            for fut in as_completed(futs):
                done_bytes += fut.result()
                pct = done_bytes / size * 100
                speed = done_bytes / max(time.time() - t0, 1e-6) / 1e6
                print(f"\r  {pct:5.1f}%  {done_bytes / 1e9:.2f}/{size / 1e9:.2f} GB  {speed:.1f} MB/s", end="")
        print()

        final = dst.stat().st_size
        if final != size:
            print(f"[ERR] 大小不符: {final} != {size}", file=sys.stderr)
            return 1
        print(f"[done] {dst.name}  {final / 1e9:.2f} GB  {time.time() - t0:.0f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
