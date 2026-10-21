"""在 UE 编辑器里采集训练数据：画面 + 位姿 + 动作 + 占据真值。

这一段脚本要**在 UE 的 Python 环境里跑**（不是系统的 Python），
入口是 MCP 的 execute_python，或 UnrealEditor-Cmd -ExecutePythonScript。

场景自己搭（不依赖 DemoProject 默认关卡有什么）：
    地面 Plane ×1、平行光 ×1、天光 ×1、移动目标 Cube ×N、相机 ×1

每条样本 = 一帧：
    frame_00000.png     384×216 RGB
    meta.jsonl          每行一条：位姿 / 动作 / 目标 3D 位置 / 栅格占据

动作 a_t 定义（相机系）：
    [forward, right, up, yaw_rate]   归一化到 [-1, 1]
    下一帧的位姿由 a_t 决定，所以"动作→观测"的因果是干净的。

⚠️ 本脚本尚未在编辑器里联调过。首次运行按下面的顺序验证：
    1) 只跑 build_scene()，确认场景搭起来
    2) 跑 10 帧，确认 PNG 落盘且不是全黑
    3) 再放量
"""

from __future__ import annotations

import json
import math
import os
import random

import unreal  # 只在 UE Python 环境里存在

# ---------------- 配置 ----------------
OUT_ROOT = "D:/zzr/data/ue_sim"
SEQ = "seq_000"
W, H = 384, 216
GRID_W, GRID_H = 48, 27          # 与现有训练管线一致（每格 8×8 像素）
N_TARGETS = 6
N_STEPS = 900                    # 30 秒 @ 30fps
DRONE_Z = 3000.0                 # 30 米（UE 单位 cm）
SPEED = 400.0                    # 每帧前进 4 米
CAM_PITCH = -45.0

random.seed(0)


def log(msg: str) -> None:
    unreal.log(f"[collect] {msg}")


# ---------------- 场景 ----------------
def clear_scene() -> None:
    sas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for a in sas.get_all_level_actors():
        label = a.get_actor_label()
        if label.startswith("SIM_"):
            sas.destroy_actor(a)


def build_scene() -> dict:
    """搭一个最小场景，返回相机和目标 Actor。"""
    sas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    eal = unreal.EditorAssetLibrary

    # 地面：Plane 默认 100×100 cm，放大 200 倍 = 200 米见方
    plane = eal.load_asset("/Engine/BasicShapes/Plane")
    ground = sas.spawn_actor_from_object(plane, unreal.Vector(0, 0, 0))
    ground.set_actor_label("SIM_Ground")
    ground.set_actor_scale3d(unreal.Vector(200, 200, 1))

    # 光
    light = sas.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 5000))
    light.set_actor_label("SIM_Sun")
    light.set_actor_rotation(unreal.Rotator(-55, 30, 0), False)

    sky = sas.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 5000))
    sky.set_actor_label("SIM_Sky")

    # 目标：立方体，随机分布在相机前方一片带状区域
    cube = eal.load_asset("/Engine/BasicShapes/Cube")
    targets = []
    for i in range(N_TARGETS):
        x = random.uniform(2000, 12000)
        y = random.uniform(-4000, 4000)
        t = sas.spawn_actor_from_object(cube, unreal.Vector(x, y, 150))
        t.set_actor_label(f"SIM_Target_{i}")
        t.set_actor_scale3d(unreal.Vector(1.5, 1.5, 1.5))   # 150cm 见方
        targets.append({
            "actor": t,
            "vel": unreal.Vector(random.uniform(-300, 300), random.uniform(-300, 300), 0),
        })

    # 相机
    cam = sas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0, 0, DRONE_Z))
    cam.set_actor_label("SIM_Camera")
    comp = cam.capture_component2d
    comp.set_editor_property("capture_every_frame", False)
    comp.set_editor_property("fov_angle", 60.0)

    rt = unreal.RenderingLibrary.create_render_target2d(
        unreal.EditorLevelLibrary.get_editor_world(), W, H, unreal.TextureRenderTargetFormat.RTF_RGBA8)
    comp.set_editor_property("texture_target", rt)

    log(f"场景就绪：{N_TARGETS} 个目标，渲染目标 {W}×{H}")
    return {"cam": cam, "comp": comp, "rt": rt, "targets": targets}


# ---------------- 动作与位姿 ----------------
def apply_action(cam, action: list[float]) -> None:
    """按动作推进相机。动作 = [forward, right, up, yaw_rate]。"""
    loc = cam.get_actor_location()
    rot = cam.get_actor_rotation()
    fwd, right, up, yaw = action
    new_loc = unreal.Vector(
        loc.x + fwd * SPEED,
        loc.y + right * SPEED * 0.5,
        max(loc.z + up * SPEED * 0.3, 500.0),
    )
    new_rot = unreal.Rotator(rot.pitch + random.uniform(-1.5, 1.5),   # 加抖动：不让位姿完全可解析
                             rot.yaw + yaw * 8.0,
                             rot.roll + random.uniform(-1.0, 1.0))
    cam.set_actor_location(new_loc, False, True)
    cam.set_actor_rotation(new_rot, False)


def move_targets(targets: list[dict]) -> None:
    for t in targets:
        a = t["actor"]
        loc = a.get_actor_location()
        v = t["vel"]
        nx = loc.x + v.x / 30.0          # 每帧 1/30 秒
        ny = loc.y + v.y / 30.0
        if abs(ny) > 5000:               # 边界反弹
            v.y *= -1
        if nx > 15000 or nx < 500:
            v.x *= -1
        a.set_actor_location(unreal.Vector(nx, ny, loc.z), False, True)


# ---------------- 占据真值 ----------------
def rasterize(targets: list[dict], cam, world) -> list[int]:
    """把目标包围盒投影到图像平面，栅格化成 0/1 网格（展平成一维存盘）。"""
    occ = [0] * (GRID_W * GRID_H)
    for t in targets:
        a = t["actor"]
        loc = a.get_actor_location()
        ok, sx, sy = unreal.GameplayStatics.project_world_location_to_screen(
            world, loc, False)
        if not ok:
            continue
        # 目标 150cm，在 384 宽、60° FOV 下约 6px；按固定 2 格半径画，近似包围盒
        gx = int(sx / W * GRID_W)
        gy = int(sy / H * GRID_H)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                x, y = gx + dx, gy + dy
                if 0 <= x < GRID_W and 0 <= y < GRID_H:
                    occ[y * GRID_W + x] = 1
    return occ


# ---------------- 主循环 ----------------
def run() -> None:
    out_dir = os.path.join(OUT_ROOT, SEQ)
    os.makedirs(out_dir, exist_ok=True)
    world = unreal.EditorLevelLibrary.get_editor_world()

    clear_scene()
    sc = build_scene()
    cam, comp, rt, targets = sc["cam"], sc["comp"], sc["rt"], sc["targets"]

    cam.set_actor_location(unreal.Vector(0, 0, DRONE_Z), False, True)
    cam.set_actor_rotation(unreal.Rotator(CAM_PITCH, 0, 0), False)

    meta_path = os.path.join(out_dir, "meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for step in range(N_STEPS):
            # 1) 采样动作（前 60% 直行，之后随机转向，制造可学的规律）
            if step < N_STEPS * 0.6:
                action = [1.0, random.uniform(-0.15, 0.15), 0.0, random.uniform(-0.1, 0.1)]
            else:
                action = [random.uniform(0.2, 1.0), random.uniform(-0.5, 0.5),
                          random.uniform(-0.2, 0.2), random.uniform(-1.0, 1.0)]

            # 2) 先推进世界，再抓这一帧（动作 → 后果）
            apply_action(cam, action)
            move_targets(targets)

            # 3) 抓图
            comp.capture_scene()
            png = os.path.join(out_dir, f"frame_{step:05d}.png")
            unreal.RenderingLibrary.export_render_target_to_file(
                world, rt, png, unreal.ImageFormat.PNG)

            rot = cam.get_actor_rotation()
            loc = cam.get_actor_location()
            rec = {
                "step": step,
                "png": os.path.basename(png),
                "pos": [loc.x, loc.y, loc.z],
                "rot": [rot.pitch, rot.yaw, rot.roll],
                "action": action,
                "targets": [[t["actor"].get_actor_location().x,
                             t["actor"].get_actor_location().y,
                             t["actor"].get_actor_location().z] for t in targets],
                "occ": rasterize(targets, cam, world),
            }
            f.write(json.dumps(rec) + "\n")

            if step % 100 == 0:
                log(f"{step}/{N_STEPS}")

    log(f"完成：{out_dir}（{N_STEPS} 帧）")


if __name__ == "__main__":
    run()
