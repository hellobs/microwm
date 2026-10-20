<h1 align="center">microwm</h1>

**面向无人机视频的微型 JEPA 世界模型——在隐空间里预测，用线性探针评测。**

一个课程尺度的研究原型，只问一个问题：*在一张消费级显卡上、在一个本来不是为它设计的数据集上，JEPA 式的隐空间动力学模型能不能训练到可用的状态？*编码器以下的一切都刻意做小。没有像素解码器，没有动作条件，也不追求任何 SOTA 数字。

[![Status](https://img.shields.io/badge/status-work%20in%20progress-d29922?style=flat-square&labelColor=1f2328)](.) [![Python](https://img.shields.io/badge/python-%E2%89%A5%203.10-3776ab?style=flat-square&labelColor=1f2328)](.) [![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A5%202.7%20%2B%20CUDA-ee4c2c?style=flat-square&labelColor=1f2328)](.) [![Dataset](https://img.shields.io/badge/dataset-VisDrone2019--MOT-7c3aed?style=flat-square&labelColor=1f2328)](.) [![License](https://img.shields.io/badge/license-Apache--2.0-3b82f6?style=flat-square&labelColor=1f2328)](LICENSE)

[English](./README.md) | **简体中文**

---

> **摘要**
>
> microwm 在短片段无人机视频上学习一个隐空间动力学模型。ViT 编码器把每帧映射成一组 patch token；带 stop-gradient 的 EMA 目标编码器提供预测目标；VICReg 的方差/协方差项防止表征坍缩；自回归预测器在从未见过真实未来的前提下，把隐状态向前滚动。评测方式是冻结特征上的线性探针，加上开环与闭环的滚动误差——**全程不训练解码器**。
>
> 它的贡献不是某个基准分数，而是一个小而可检视的「预测式表征」实例：坍缩、滚动漂移、patch 粒度这三类失败模式都被摆到台面上并量化。

**范围**

| 做了什么 | 明确不做 |
|---|---|
| 隐空间预测（`1 - cos`），不重建像素 | 动作条件动力学——数据里没有控制信号 |
| EMA 目标编码器 + stop-gradient + VICReg 正则 | 掩码图像建模（这里是时间外推，不是补洞） |
| 固定时域上的自回归滚动 | 真正的鸟瞰图；没有相机标定参数 |
| 冻结特征上的线性探针评测 | 检测/跟踪精度；标注框只用来做占据目标 |
| 以开环/闭环误差为核心证据 | 大规模预训练；模型刻意做得很小 |

---

## 目录

* [1\. 当前状态](#1-当前状态)
* [2\. 设计取舍](#2-设计取舍)
* [3\. 仓库结构](#3-仓库结构)
* [4\. 快速开始](#4-快速开始)
* [5\. 路线图](#5-路线图)
* [6\. 局限](#6-局限)
* [7\. 数据与许可](#7-数据与许可)

---

## 1. 当前状态

代码写完 和 验证通过 是两件事，这张表分开记。

| 模块 | 代码 | 已验证 |
|---|---|---|
| 环境自检 `scripts/check_env.py` | 完成 | 待验证 |
| 数据下载器 `scripts/download_data.py`（分块并行、可续传） | 完成 | 验证集已下载并解压 |
| 预处理 → 每序列 `npy` + 归一化标注 | 完成 | 待跑全量 |
| 占据栅格数据集（memmap + clip 采样） | 完成 | 待验证 |
| ViT 编码器 + EMA 目标编码器 + VICReg | 计划中 | — |
| 自回归隐空间预测器 | 计划中 | — |
| 线性探针，persistence / 线性外推 / ConvLSTM 基线 | 计划中 | — |
| 训练与评测入口 | 计划中 | — |
| 论文（LaTeX） | 计划中 | — |

## 2. 设计取舍

三个决定影响了其余所有部分，每个都带着代价。

**预测表征，不预测像素。** 损失是隐空间里的 `1 - cos(pred, target)`。用 500 万参数的模型在笔记本显卡上重建 192×108 的 RGB，结果一定是糊的，而"糊"证明不了任何事。代价是：拿不出一张漂亮的预测帧，所有定性图只能是占据栅格和特征图——不好看，但诚实。

**两重独立的防坍缩机制。** 带 stop-gradient 的 EMA 目标编码器消除了「编码器输出常数」这个平凡解；VICReg 的方差与协方差项把"不坍缩"从期望变成显式目标。代价是多两个超参（`0.04` / `0.01`，直接取自 VICReg 原文，未重新调），以及一个使编码器显存翻倍的教师网络。

**用占据栅格代替 BEV。** VisDrone 没有提供内参，且各序列俯角不一，硬做逆透视变换会引入一个无法量化的误差项。所以改为在图像平面离散化，并在论文里写明。代价是：这里的"占据"指图像平面占据，靠近地平线的物体会被压缩——这条写进讨论章，不藏。

## 3. 仓库结构

```
configs/          实验配置（smoke / base）
scripts/          下载器、环境自检、patch 可视化
src/microwm/
  data/           预处理、占据栅格、clip 数据集
LICENSE
README.md         English
README_zh.md      本文件
```

训练、评测与论文目录会随阶段推进补齐——上面只列已存在的东西。

## 4. 快速开始

```bash
python -m venv .venv
# Windows: .venv/Scripts/activate    macOS/Linux: source .venv/bin/activate

# Blackwell 架构显卡需要 cu128 轮子；较老的卡用默认源即可
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python scripts/check_env.py                       # 确认 CUDA 与计算能力
python scripts/download_data.py --subset val      # 分块并行，可续传
python -m microwm.data.preprocess --root <解压目录> --split val
```

本仓库不再分发数据集。官方链接是 Google Drive 与百度网盘，因此下载走镜像源；见[数据与许可](#7-数据与许可)。

## 5. 路线图

| 阶段 | 交付物 | 进入下一阶段前必须过的一关 |
|---|---|---|
| 数据 | 预处理好的 clip + 四张数据统计图 | persistence 基线能拿到不像 0 的占据 IoU——否则是管线错了，不是模型不行 |
| 表征 | 训练好的编码器、loss 曲线、特征图 | 特征方差没有塌到 0，即未坍缩 |
| 动力学 | 闭环滚动、误差-时域曲线 | 最后一个时域上隐空间余弦仍有意义 |
| 评测 | 主对比表、消融、线性探针 | 每个数字都能由一条命令复现 |
| 成文 | 论文、图表、发布 | 每条结论都能追溯到某个实验 |

## 6. 局限

- **没有动作信号。** 模型预测"接下来会发生什么"，而不是"如果怎样会发生什么"。要做成动作条件模型，需要位姿或控制数据，而这个数据集不提供。
- **小数据区间。** 自监督表征学习通常在比这大几个数量级的规模上被验证。
- **patch 粒度。** 该数据集里很多目标小于单个 patch，空间精度因此受构造限制，线性探针预计是最薄的一环。
- **单数据集、单域。** 不声称任何跨数据集迁移能力。

## 7. 数据与许可

基于 [VisDrone2019-MOT](https://github.com/VisDrone/VisDrone-Dataset)，由天津大学 AISKYEYE 团队采集。数据集按其自身条款使用，不在本仓库中再分发；仓库内只保存派生统计与本代码产出的模型权重。

代码以 [Apache-2.0](LICENSE) 发布。
