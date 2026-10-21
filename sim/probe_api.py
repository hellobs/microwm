"""先别跑采集，先验 API 在不在。

UE Python 的 API 名字跨版本会变，写错一个整段就废。
这个脚本只做一件事：把采集脚本用到的每个 API 逐个试一遍，报告哪些存在。

在 UE 里跑：execute_python(code=open('D:/zzr/microwm/sim/probe_api.py').read())
"""

import unreal

CHECKS = [
    # (说明, 表达式, 是否必需)
    ("EditorActorSubsystem", lambda: unreal.get_editor_subsystem(unreal.EditorActorSubsystem), True),
    ("EditorAssetLibrary", lambda: unreal.EditorAssetLibrary.list_assets("/Engine/BasicShapes"), True),
    ("BasicShapes/Cube", lambda: unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cube"), True),
    ("BasicShapes/Plane", lambda: unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Plane"), True),
    ("SceneCapture2D 类", lambda: unreal.SceneCapture2D, True),
    ("create_render_target2d", lambda: unreal.RenderingLibrary.create_render_target2d, True),
    ("export_render_target_to_file", lambda: unreal.RenderingLibrary.export_render_target_to_file, True),
    ("ImageFormat.PNG", lambda: unreal.ImageFormat.PNG, True),
    ("project_world_location_to_screen", lambda: unreal.GameplayStatics.project_world_location_to_screen, True),
    ("get_editor_world", lambda: unreal.EditorLevelLibrary.get_editor_world(), True),
    ("DirectionalLight 类", lambda: unreal.DirectionalLight, True),
    ("SkyLight 类", lambda: unreal.SkyLight, True),
    ("Rotator(pitch,yaw,roll)", lambda: unreal.Rotator(-45, 0, 0), True),
    # 可选：拿不到就换实现
    ("RenderTargetFormat", lambda: unreal.TextureRenderTargetFormat.RTF_RGBA8, False),
]

ok = fail = 0
for name, fn, required in CHECKS:
    try:
        v = fn()
        unreal.log(f"[probe] OK    {name}  -> {type(v).__name__}")
        ok += 1
    except Exception as e:  # noqa: BLE001
        tag = "MISS " if required else "WARN "
        unreal.log(f"[probe] {tag} {name}  -> {e}")
        if required:
            fail += 1

unreal.log(f"[probe] 合计：通过 {ok}，缺失 {fail}")
unreal.log("[probe] 缺失为 0 才继续跑采集脚本")

# 额外：确认捕获组件能拿到
try:
    sas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    cam = sas.spawn_actor_from_class(unreal.SceneCapture2D, unreal.Vector(0, 0, 3000))
    comp = cam.capture_component2d
    unreal.log(f"[probe] OK    capture_component2d -> {type(comp).__name__}")
    unreal.log(f"[probe] fov_angle 当前值 = {comp.get_editor_property('fov_angle')}")
    sas.destroy_actor(cam)
except Exception as e:  # noqa: BLE001
    unreal.log(f"[probe] MISS  capture_component2d -> {e}")
