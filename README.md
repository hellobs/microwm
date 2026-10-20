# MicroJEPA · 微型视觉世界模型

在 **VisDrone2019-MOT** 上训练一个 4–5M 参数的 JEPA 式世界模型：用 ViT 编码，
EMA 目标编码器 + VICReg 正则约束表征，自回归预测器在隐空间里滚动预测未来，
并以**线性探针**评估表征质量。

课程：《小白学人工智能》数据分析大作业
题目：微型视觉世界模型——基于 VisDrone 的无人机场景时空动态预测与异常分析

## 状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | 环境与数据 | 进行中 |
| 2 | 预处理与数据洞察 | 待开始 |
| 3 | 表征训练 | 待开始 |
| 4 | 闭环滚动与基线 | 待开始 |
| 5 | 实验矩阵与论文 | 待开始 |

## 环境

```bash
python -m venv .venv
.venv/Scripts/pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python scripts/check_env.py     # 期望 capability = (12, 0)
```

本机：RTX 5060 Laptop 8GB · Blackwell sm_120 · Python 3.13 · torch 2.9.1+cu128

## 数据

VisDrone2019-MOT（官方源是 Google Drive / 百度网盘，国内不可达，走 hf-mirror 镜像）：

```bash
python scripts/download_data.py --subset val      # 1.6 GB
python scripts/download_data.py --subset train --jobs 8   # 8.1 GB
```

解压到 `D:/zzr/data/visdrone/`（不进仓库）。

## 目录

```
scripts/    下载、自检、预处理、画图
src/microwm 模型与训练代码
configs/    实验配置
outputs/    checkpoint、日志、图（不入库）
paper/      LaTeX 正文
```

## 设计要点

- **不做像素重建**：损失在表征空间计算（`1 - cos`），无解码器。
- **防坍缩**：EMA 目标编码器 + stop-grad，另加 VICReg 方差/协方差正则。
- **闭环滚动**：训练用 teacher forcing，评测同时报开环与自回归两种误差——
  后者是"世界模型能滚多远"的唯一证据。
- **占据栅格**：VisDrone 无相机标定，不做 IPM，用图像平面 48×48 栅格近似，
  讨论章说明与严格 BEV 的差距。
