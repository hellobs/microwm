"""环境自检：确认 torch 能看到 Blackwell GPU。

期望输出第二行为 (12, 0) —— sm_120。若不是，说明装错了轮子，
不用往下走，先卸了重装 cu128 版本。
"""

import platform
import sys


def main() -> int:
    print(f"python  {platform.python_version()}  ({sys.executable})")

    try:
        import torch
    except ImportError:
        print("torch   未安装")
        return 1

    print(f"torch   {torch.__version__}")
    ok = torch.cuda.is_available()
    print(f"cuda    {ok}")

    if not ok:
        print("GPU 不可用：检查驱动，或确认装的是 cu128 轮子")
        return 1

    cap = torch.cuda.get_device_capability()
    print(f"device  {torch.cuda.get_device_name(0)}")
    print(f"cap     {cap}  (Blackwell sm_120 应为 (12, 0))")
    print(f"vram    {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

    # 真正跑一次 kernel，确认不是"能看见但算不了"
    a = torch.randn(512, 512, device="cuda")
    b = a @ a
    torch.cuda.synchronize()
    print(f"matmul  ok  {tuple(b.shape)}  {b.sum().item():.3f}")

    ok_cap = cap == (12, 0)
    print("\n结论：" + ("环境正确，可以开工" if ok_cap else "capability 不是 (12,0)，轮子可能装错了"))
    return 0 if ok_cap else 2


if __name__ == "__main__":
    raise SystemExit(main())
