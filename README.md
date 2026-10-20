<h1 align="center">microwm</h1>

**A micro-scale JEPA world model for drone video — predict in latent space, evaluate with a linear probe.**

A course-scale research prototype that asks one narrow question: *can a JEPA-style latent dynamics model be trained to a usable state on a single consumer GPU, on a dataset that was never designed for it?* Everything below the encoder is deliberately small. There is no pixel decoder, no action conditioning, and no attempt at state of the art.

[![Status](https://img.shields.io/badge/status-work%20in%20progress-d29922?style=flat-square&labelColor=1f2328)](.) [![Python](https://img.shields.io/badge/python-%E2%89%A5%203.10-3776ab?style=flat-square&labelColor=1f2328)](.) [![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A5%202.7%20%2B%20CUDA-ee4c2c?style=flat-square&labelColor=1f2328)](.) [![Dataset](https://img.shields.io/badge/dataset-VisDrone2019--MOT-7c3aed?style=flat-square&labelColor=1f2328)](.) [![License](https://img.shields.io/badge/license-Apache--2.0-3b82f6?style=flat-square&labelColor=1f2328)](LICENSE)

**English** | [简体中文](./README_zh.md)

---

> **Abstract**
>
> microwm learns a latent dynamics model over short drone-video clips. A ViT context encoder maps each frame to a set of patch tokens; an EMA target encoder with stop-gradient provides the prediction target; a VICReg variance/covariance term keeps the representation from collapsing; an autoregressive predictor rolls the latent state forward without ever seeing the ground-truth future. Evaluation is a linear probe on frozen features plus open-loop and closed-loop rollout error — **no decoder is trained at any point**.
>
> The contribution is not a benchmark number. It is a small, inspectable instance of the predictive-representation paradigm, with every failure mode (collapse, rollout drift, patch granularity) made visible and measured.

**Scope**

| Covered | Not covered |
|---|---|
| Latent-space prediction (`1 - cos`), no pixel reconstruction | Action-conditioned dynamics — the dataset carries no control signal |
| EMA target encoder + stop-gradient + VICReg regularisation | Masked-image modelling (this is temporal extrapolation, not inpainting) |
| Autoregressive rollout over a fixed horizon | True bird's-eye-view; no camera calibration is available |
| Linear-probe evaluation of frozen features | Detection or tracking accuracy; boxes are only used as occupancy targets |
| Open-loop vs closed-loop error as the core evidence | Large-scale pretraining; the model is intentionally tiny |

---

## Table of Contents

* [1\. Status](#1-status)
* [2\. Design notes](#2-design-notes)
* [3\. Repository layout](#3-repository-layout)
* [4\. Quick start](#4-quick-start)
* [5\. Roadmap](#5-roadmap)
* [6\. Limitations](#6-limitations)
* [7\. Data & license](#7-data--license)

---

## 1. Status

Honest bookkeeping: *code written* and *verified to run* are different things, and the table tracks both.

| Component | Code | Verified |
|---|---|---|
| Environment check (`scripts/check_env.py`) | done | `torch 2.9.1+cu128`, compute capability `(12, 0)`, 8 GiB |
| Dataset downloader (`scripts/download_data.py`) — parallel ranges, resumable | done | validation and training splits fetched |
| Preprocessing → per-sequence `npy` + normalised boxes | done | pending full run |
| Occupancy-grid dataset (memmap, clip sampler) | done | pending |
| Data statistics (four-panel figure) | done | run on the validation split |
| ViT encoder + EMA target encoder + VICReg | done | forward/backward and rollout smoke-tested |
| Autoregressive latent predictor | done | smoke-tested |
| Linear probe, persistence / linear-extrapolation / ConvLSTM baselines | probe head written, baselines planned | — |
| Training and evaluation entry points | planned | — |
| Paper (LaTeX) | planned | — |

## 2. Design notes

Three choices that shape everything else, each with a cost attached.

**Predict representations, not pixels.** The loss is `1 - cos(pred, target)` in latent space. Reconstructing 192×108 RGB from a 5 M-parameter model on a laptop GPU produces blur, and blur is not evidence of anything. The cost: we cannot show a pretty predicted frame, so all qualitative figures are occupancy grids and feature maps — less impressive looking, more honest.

**Two independent defences against collapse.** An EMA target encoder with stop-gradient removes the trivial solution where the encoder outputs a constant; VICReg's variance and covariance terms make non-collapse an explicit objective rather than a hope. The cost: two extra hyper-parameters (`0.04` / `0.01`, taken from the VICReg paper and not re-tuned), and a teacher network that doubles encoder memory.

**Occupancy grid instead of BEV.** VisDrone ships no intrinsics, and camera pitch varies between sequences, so a proper inverse perspective mapping would introduce an error term we cannot quantify. We discretise the image plane instead and say so in the paper. The cost: "occupancy" here means *image-plane* occupancy, and objects near the horizon are compressed — a limitation stated in the discussion, not hidden.

## 3. Repository layout

```
configs/          experiment configs (smoke / base)
scripts/          downloader, environment check, patch visualiser
src/microwm/
  data/           preprocessing, occupancy rasteriser, clip dataset
LICENSE
README.md         this file
README_zh.md      简体中文
```

Training, evaluation and paper directories are added as each phase lands — the layout above only lists what exists.

## 4. Quick start

```bash
python -m venv .venv
# Windows: .venv/Scripts/activate    macOS/Linux: source .venv/bin/activate

# Blackwell-class GPUs need the cu128 wheel; older cards can use the default index
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python scripts/check_env.py                       # confirms CUDA + compute capability
python scripts/download_data.py --subset val      # resumable, parallel-range download
python -m microwm.data.preprocess --root <extracted-dir> --split val
```

The dataset is not redistributed here. Downloads go through a mirror because the official links are Google Drive and Baidu Netdisk; see [Data & license](#7-data--license).

## 5. Roadmap

| Phase | Deliverable | Gate before moving on |
|---|---|---|
| Data | Preprocessed clips + four data-statistics figures | Persistence baseline reaches a non-trivial occupancy IoU — otherwise the pipeline is wrong, not the model |
| Representation | Trained encoder, loss curves, feature maps | Feature variance stays above zero, i.e. no collapse |
| Dynamics | Closed-loop rollout, error-vs-horizon curve | Latent cosine similarity still meaningful at the last horizon |
| Evaluation | Main comparison table, ablations, linear probe | Every number reproducible from a single command |
| Write-up | Paper, figures, release | Each claim traceable to an experiment |

## 6. Limitations

- **No action signal.** The model predicts what happens next, not what happens *if*. Making it action-conditioned requires pose or control data that this dataset does not provide.
- **Small-data regime.** Self-supervised representation learning is usually demonstrated at a scale several orders of magnitude larger than this.
- **Patch granularity.** Objects in this dataset are frequently smaller than a single patch; spatial precision is therefore limited by construction, and the linear probe is expected to be the weakest link.
- **Single dataset, single domain.** No cross-dataset transfer is claimed.

## 7. Data & license

Built on [VisDrone2019-MOT](https://github.com/VisDrone/VisDrone-Dataset), collected by the AISKYEYE team at Tianjin University. The dataset is used under its own terms and is not redistributed in this repository; only derived statistics and model weights produced by this code are stored here.

Code is released under the [Apache-2.0 License](LICENSE).
