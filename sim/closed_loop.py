"""闭环对照实验：直飞 / 真值避障 / 世界模型闭环。

三种模式跑同一组种子与场景，统计成功率、碰撞率、最小净空。

    python sim/closed_loop.py --mode straight --episodes 10 --out D:/zzr/data/cl_straight
    python sim/closed_loop.py --mode truth    --episodes 10 --out D:/zzr/data/cl_truth
    python sim/closed_loop.py --mode model    --episodes 10 --out D:/zzr/data/cl_model \
        --ckpt D:/zzr/microwm/outputs/dl_act/model.pt

model 模式会自己拉起 Tools/world_model_service.py，跑完关掉。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_dronelab import run_episode  # noqa: E402

SERVICE = Path(r"D:\zzr\UE-DroneSim\Tools\world_model_service.py")
PYTHON = Path(r"D:\zzr\microwm\.venv\Scripts\python.exe")


def summarise_episode(ep_dir: Path) -> dict:
    frames = [json.loads(x) for x in (ep_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    events = []
    ev = ep_dir / "events.jsonl"
    if ev.exists():
        events = [json.loads(x) for x in ev.read_text(encoding="utf-8").splitlines() if x.strip()]
    end = next((e for e in events if e.get("type") == "episode_end"), {})
    result = end.get("result") or (frames[-1].get("result") if frames else "?")
    clear = min((f.get("minimum_clearance_m", 99) for f in frames), default=99)
    return {
        "frames": len(frames),
        "result": result,
        "collisions": int(frames[-1].get("collisions", 0)) if frames else 0,
        "min_clearance_m": round(float(clear), 3),
        "duration_s": round(float(frames[-1].get("sim_time_s", 0)), 2) if frames else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["straight", "truth", "model"], required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--scenario", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=500)
    ap.add_argument("--hz", type=float, default=10)
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--port", type=int, default=8799)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    svc = None
    if args.mode == "model":
        if not args.ckpt:
            raise SystemExit("model 模式需要 --ckpt")
        svc = subprocess.Popen(
            [str(PYTHON), str(SERVICE), "--ckpt", args.ckpt, "--port", str(args.port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
        time.sleep(12)   # 等权重载入
        print(f"[closed] 服务已启动 pid={svc.pid}", flush=True)

    rows = []
    try:
        for i in range(args.episodes):
            seed = args.seed0 + i
            r = run_episode(
                out, args.scenario, seed, args.seconds, args.hz,
                avoid=(args.mode != "straight"),
                bridge=(args.mode == "model"),
                tag_suffix="" if args.mode != "model" else "_wm",
            )
            for ep in r.get("episodes", []):
                d = out / r["tag"] / ep["dir"]
                rows.append({"seed": seed, **summarise_episode(d)})
    finally:
        if svc is not None:
            svc.terminate()

    n = len(rows)
    ok = sum(1 for r in rows if r["result"] == "success")
    hit = sum(1 for r in rows if r["result"] == "collision")
    oob = sum(1 for r in rows if r["result"] == "out_of_bounds")
    clear = [r["min_clearance_m"] for r in rows if r["min_clearance_m"] < 99]
    report = {
        "mode": args.mode,
        "scenario": args.scenario,
        "episodes": n,
        "success": ok,
        "collision": hit,
        "out_of_bounds": oob,
        "other": n - ok - hit - oob,
        "success_rate": round(ok / n, 3) if n else 0.0,
        "collision_rate": round(hit / n, 3) if n else 0.0,
        "min_clearance_mean": round(sum(clear) / len(clear), 3) if clear else None,
        "rows": rows,
    }
    (out / f"closed_{args.mode}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
