"""批量采集 DroneLab 数据集。

调用 UE 的 DroneSim 工程，逐个种子/场景跑一个回合，回合结束自动退出。
输出目录结构（每个回合一个子目录）：

    <out>/<scenario>_s<seed>/
        metadata.json     相机内参、坐标约定、动作语义
        frames.jsonl      每帧状态/动作/障碍真值
        events.jsonl      碰撞、越界、回合结束事件
        summary.json      帧数、图像数、桥接响应数
        rgb/000000.png    第一视角画面

用法：
    python sim/collect_dronelab.py --episodes 12 --scenario 2 --out D:/zzr/data/dronelab

UE 侧依赖的命令行开关（见 Source/DroneSim/SimpleDronePawn.cpp）：
    -DroneAuto -DroneRecord -DroneExitOnFinish -DroneScenario -DroneSeed
    -DroneOutputDir -DroneNoAvoid -DroneSampleHz
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ENGINE = Path(r"D:\Program Files\Epic Games\UE_5.7")
EDITOR = ENGINE / "Engine" / "Binaries" / "Win64" / "UnrealEditor.exe"
PROJECT = Path(r"D:\zzr\UE-DroneSim\DroneSim.uproject")
MAP = "/Game/Maps/DroneLab"


def run_episode(
    out_root: Path,
    scenario: int,
    seed: int,
    seconds: float,
    hz: float,
    avoid: bool,
    no_render: bool = False,
    bridge: bool = False,
    tag_suffix: str = "",
) -> dict:
    """跑一个回合，返回从 summary.json 读到的小结。"""
    tag = f"s{scenario}_seed{seed}" + ("" if avoid else "_noavoid") + tag_suffix
    target = out_root / tag
    target.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(EDITOR),
        str(PROJECT),
        MAP,
        "-game",
        "-windowed",
        "-ResX=960",
        "-ResY=540",
        "-NoSplash",
        "-NoSound",
        "-DroneAuto",
        "-DroneRecord",
        "-DroneExitOnFinish",
        f"-DroneScenario={scenario}",
        f"-DroneSeed={seed}",
        f"-DroneOutputDir={target.as_posix()}",
        f"-DroneSampleHz={hz}",
    ]
    if not avoid:
        cmd.append("-DroneNoAvoid")
    if bridge:
        cmd.append("-DroneBridge")
    if no_render:
        cmd.append("-nullrhi")

    print(f"[collect] {tag} …", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 120)

    dirs = sorted(p for p in target.iterdir() if p.is_dir() and p.name.startswith("20"))
    summary: dict = {"tag": tag, "returncode": proc.returncode, "episodes": []}
    for d in dirs:
        s = d / "summary.json"
        if s.exists():
            info = json.loads(s.read_text(encoding="utf-8"))
            info["dir"] = d.name
            summary["episodes"].append(info)
    summary["wall_s"] = round(time.time() - started, 1)
    if not summary["episodes"]:
        tail = (proc.stderr or proc.stdout or "")[-600:]
        summary["error"] = tail
    print(
        f"[collect] {tag} rc={proc.returncode} "
        f"episodes={len(summary['episodes'])} wall={summary['wall_s']}s",
        flush=True,
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"D:\zzr\data\dronelab")
    ap.add_argument("--episodes", type=int, default=8, help="总回合数")
    ap.add_argument("--scenario", type=int, default=2, help="场景编号 0-3，-1 表示轮换")
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=75.0, help="单回合墙钟上限")
    ap.add_argument("--hz", type=float, default=10.0, help="图像采样率")
    ap.add_argument("--no-avoid", action="store_true", help="关闭真值避障（对照组）")
    ap.add_argument("--report", default=None, help="写汇总 json 的路径")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(args.episodes):
        scenario = (i % 4) if args.scenario < 0 else args.scenario
        seed = args.seed0 + i
        try:
            results.append(run_episode(out_root, scenario, seed, args.seconds, args.hz, not args.no_avoid))
        except subprocess.TimeoutExpired:
            print(f"[collect] seed={seed} 超时，跳过", flush=True)
            results.append({"tag": f"s{scenario}_seed{seed}", "timeout": True})

    total_frames = sum(e.get("frames", 0) for r in results for e in r.get("episodes", []))
    total_images = sum(e.get("images", 0) for r in results for e in r.get("episodes", []))
    report = {
        "out": str(out_root),
        "rounds": results,
        "total_frames": total_frames,
        "total_images": total_images,
    }
    path = Path(args.report) if args.report else out_root / "collect_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[collect] 共 {total_frames} 帧 / {total_images} 张图 → {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
