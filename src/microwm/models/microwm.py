"""MicroJEPA：ViT 编码 + EMA 教师 + 自回归预测器 + 轻量探针。

三条约束决定了这里的每个设计：
  1. 预测发生在隐空间（1 - cos），没有像素解码器；
  2. 防坍缩靠两重机制 —— EMA + stop-grad，以及 VICReg 的方差/协方差项；
  3. 教师只看未来帧，学生只看过去帧，二者解耦。

张量形状（batch=B, 帧数=T, 边长=192×108, patch=12）：
    输入      [B, T, 3, 108, 192]
    编码      [B, T, 144, 192]        144 = 16 × 9
    预测      [B, 144, 192]
    探针输出  [B, 1, 27, 48]
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                      # [B, D, gh, gw]
        return x.flatten(2).transpose(1, 2)   # [B, N, D]


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class Encoder(nn.Module):
    """单帧空间编码器，时间维度由外部展开。"""

    def __init__(self, cfg: dict, n_tokens: int) -> None:
        super().__init__()
        dim, depth, patch = cfg["dim"], cfg["depth"], cfg["patch"]
        self.patch = patch
        self.embed = PatchEmbed(dim, patch)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(dim, cfg["heads"], cfg.get("mlp_ratio", 4.0)) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _, _, _ = x.shape
        z = self.embed(x.flatten(0, 1))       # [B*T, N, D]
        z = z + self.pos
        for blk in self.blocks:
            z = blk(z)
        return self.norm(z).view(b, t, z.shape[1], self.dim)


class Predictor(nn.Module):
    """在时间轴上自回归：看第 0..t 帧（以及每帧的动作），输出第 t+1 帧的 token。"""

    def __init__(self, cfg: dict, n_tokens: int) -> None:
        super().__init__()
        dim = cfg["dim"]
        self.dim = dim
        self.n = n_tokens
        self.temporal = nn.Parameter(torch.zeros(1, 64, 1, dim))  # 最多 64 帧
        nn.init.trunc_normal_(self.temporal, std=0.02)
        self.in_proj = nn.Linear(dim, dim)
        # 动作条件：把 4 维动作（机体速度 3 + 偏航率 1）投影成与 token 同维的偏置，
        # 加在该帧所有 token 上。逐帧相加而不是全局拼接，预测器才能区分「哪一步用了什么动作」。
        self.use_actions = cfg.get("use_actions", True)
        self.act_dim = cfg.get("act_dim", 4)
        if self.use_actions:
            self.act_proj = nn.Sequential(
                nn.Linear(self.act_dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
        self.blocks = nn.ModuleList(
            [Block(dim, cfg["heads"], cfg.get("mlp_ratio", 4.0)) for _ in range(cfg["pred_depth"])]
        )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)

    @staticmethod
    def causal_mask(t: int, n: int, device: torch.device) -> torch.Tensor:
        """块因果：第 i 帧的 token 只能看到 0..i 帧（含自己）。True = 屏蔽。"""
        frame = torch.arange(t, device=device).repeat_interleave(n)
        allowed = frame[None, :] <= frame[:, None]     # [T*N, T*N]
        return ~allowed

    def forward(self, z: torch.Tensor, act: torch.Tensor | None = None) -> torch.Tensor:
        b, t, n, d = z.shape
        x = self.in_proj(z) + self.temporal[:, :t]
        if self.use_actions and act is not None:
            a = self.act_proj(act.to(x.dtype))             # [B, T, D]
            x = x + a.unsqueeze(2)                         # 广播到该帧全部 token
        x = x.flatten(1, 2)                            # [B, T*N, D]
        mask = self.causal_mask(t, n, z.device)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        x = self.norm(x).view(b, t, n, d)
        return self.out(x[:, -1])                      # 最后一帧的 token → 预测下一帧


class Probe(nn.Module):
    """轻量探针：把 token 特征还原成占据图。编码器冻结，只训这个头。"""

    def __init__(self, dim: int, grid_h: int, grid_w: int) -> None:
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.head = nn.Sequential(
            nn.Conv2d(dim, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, tokens: torch.Tensor, tok_h: int, tok_w: int) -> torch.Tensor:
        b, n, d = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, d, tok_h, tok_w)   # [B, D, gh, gw]
        x = self.head(x)
        return F.interpolate(x, size=(self.grid_h, self.grid_w), mode="bilinear", align_corners=False)


class MicroJEPA(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        c = cfg.get("model", cfg)
        d = cfg.get("data", {})
        self.cfg = cfg
        dim, patch = c["dim"], c["patch"]

        img_h = d.get("img_h", 108)
        img_w = d.get("img_w", 192)
        self.tok_h, self.tok_w = img_h // patch, img_w // patch
        self.n_tokens = self.tok_h * self.tok_w

        self.encoder = Encoder(c, self.n_tokens)
        self.teacher = copy.deepcopy(self.encoder)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.predictor = Predictor(c, self.n_tokens)
        self.probe = Probe(dim, d.get("grid_h", 27), d.get("grid_w", 48))
        self.tau = c.get("ema_tau", 0.996)

    # ---------- 基础 ----------

    @torch.no_grad()
    def update_teacher(self, tau: float | None = None) -> None:
        tau = self.tau if tau is None else tau
        for ps, pt in zip(self.encoder.parameters(), self.teacher.parameters()):
            pt.data.mul_(tau).add_(ps.data, alpha=1 - tau)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    # ---------- 损失 ----------

    @staticmethod
    def _vicreg(z: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
        z = z.reshape(-1, z.shape[-1])
        std = torch.sqrt(z.var(dim=0) + eps)
        loss_var = F.relu(1.0 - std).mean()
        zc = z - z.mean(dim=0)
        cov = zc.T @ zc / (z.shape[0] - 1)
        d = cov.shape[0]
        off = cov - torch.diag(torch.diag(cov))
        loss_cov = off.pow(2).sum() / d
        return loss_var, loss_cov

    def forward(self, past: torch.Tensor, future: torch.Tensor,
                act_past: torch.Tensor | None = None,
                act_future: torch.Tensor | None = None) -> dict:
        """teacher forcing：每一步都拿真实的上一帧表征作输入。"""
        b, t_in, _, h, w = past.shape
        k = future.shape[1]

        z_past = self.encode(past)                       # [B, T_in, N, D]
        with torch.no_grad():
            z_future = self.teacher(future)              # 目标，不回传

        loss_pred = past.new_zeros(())
        preds = []
        for step in range(k):
            inp = torch.cat([z_past, z_future[:, :step]], dim=1)
            if act_past is not None and act_future is not None:
                act_in = torch.cat([act_past, act_future[:, :step]], dim=1)
            else:
                act_in = None
            p = self.predictor(inp, act_in)              # [B, N, D]
            preds.append(p)
            tgt = z_future[:, step].detach()
            loss_pred = loss_pred + (1 - F.cosine_similarity(p, tgt, dim=-1)).mean()
        loss_pred = loss_pred / k

        z_all = self.encode(torch.cat([past, future], dim=1))
        l_var, l_cov = self._vicreg(z_all)

        c = self.cfg.get("model", self.cfg)
        lw = c.get("vicreg_var", 0.04)
        cw = c.get("vicreg_cov", 0.01)
        total = loss_pred + lw * l_var + cw * l_cov

        return {
            "loss": total,
            "pred": loss_pred.detach(),
            "var": l_var.detach(),
            "cov": l_cov.detach(),
            "preds": preds,
        }

    # ---------- 推理：闭环滚动 ----------

    def rollout_latents(self, past: torch.Tensor, horizon: int = 5,
                        act_past: torch.Tensor | None = None,
                        act_future: torch.Tensor | None = None) -> torch.Tensor:
        """闭环滚动的隐状态序列 [B, horizon, N, D]，不经过探针。

        act_future 是候选动作序列：闭环里每一步的输入都包含「这一步准备执行的动作」，
        所以模型可以回答「如果这么飞，接下来会看到什么」。
        """
        z = self.encode(past)
        acts = act_past
        outs = []
        for step in range(horizon):
            nxt = self.predictor(z, acts)
            outs.append(nxt)
            z = torch.cat([z, nxt.unsqueeze(1)], dim=1)
            if act_future is not None:
                nxt_act = act_future[:, step:step + 1]
                acts = nxt_act if acts is None else torch.cat([acts, nxt_act], dim=1)
        return torch.stack(outs, dim=1)

    def rollout(self, past: torch.Tensor, horizon: int = 5, closed_loop: bool = True,
                future: torch.Tensor | None = None,
                act_past: torch.Tensor | None = None,
                act_future: torch.Tensor | None = None) -> torch.Tensor:
        """闭环：把自己的预测拼回输入继续预测，全程不看真实未来。

        返回 [B, horizon, 1, grid_h, grid_w] 的占据概率。
        closed_loop=False 时必须提供 future，用于对照（teacher forcing）。
        """
        if closed_loop or future is None:
            z = self.encode(past)
            acts = act_past
            outs = []
            for step in range(horizon):
                nxt = self.predictor(z, acts)
                outs.append(self.probe(nxt, self.tok_h, self.tok_w).sigmoid())
                z = torch.cat([z, nxt.unsqueeze(1)], dim=1)
                if act_future is not None:
                    nxt_act = act_future[:, step:step + 1]
                    acts = nxt_act if acts is None else torch.cat([acts, nxt_act], dim=1)
            return torch.stack(outs, dim=1)

        z = self.encode(past)
        acts = act_past
        outs = []
        for step in range(horizon):
            nxt = self.predictor(z, acts)
            outs.append(self.probe(nxt, self.tok_h, self.tok_w).sigmoid())
            with torch.no_grad():
                z = torch.cat([z, self.teacher(future[:, step:step + 1])], dim=1)
            if act_future is not None:
                nxt_act = act_future[:, step:step + 1]
                acts = nxt_act if acts is None else torch.cat([acts, nxt_act], dim=1)
        return torch.stack(outs, dim=1)

