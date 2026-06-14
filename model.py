import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

"""
  - Benefit-Gated Shared Mamba, a compute-adaptive HAR framework
  - Author: JunYoung Park and Myung-Kyu Yi
"""

class MiniMambaBlock(nn.Module):
    """Lightweight Mamba-style temporal block"""
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        conv_kernel: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv_kernel = conv_kernel

        self.norm = nn.LayerNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2)

        self.depthwise_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=hidden_dim,
            bias=True,
        )

        self.dt_proj = nn.Linear(hidden_dim, hidden_dim)
        self.b_proj = nn.Linear(hidden_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)

        self.A_log = nn.Parameter(torch.zeros(hidden_dim))
        self.D = nn.Parameter(torch.ones(hidden_dim))

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def selective_scan(self, u: torch.Tensor) -> torch.Tensor:
        B, T, H = u.shape

        dt = F.softplus(self.dt_proj(u)) + 1e-4
        b_t = torch.tanh(self.b_proj(u))
        c_t = torch.tanh(self.c_proj(u))

        A = torch.exp(self.A_log).view(1, H)
        state = torch.zeros(B, H, device=u.device, dtype=u.dtype)
        outputs = []

        for t in range(T):
            dt_cur = dt[:, t, :]
            u_cur = u[:, t, :]
            b_cur = b_t[:, t, :]
            c_cur = c_t[:, t, :]

            a_bar = torch.exp(-dt_cur * A)
            b_bar = (1.0 - a_bar) * b_cur

            state = a_bar * state + b_bar * u_cur
            y_cur = c_cur * state + self.D.view(1, H) * u_cur

            outputs.append(y_cur)

        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)

        u = self.depthwise_conv(u.transpose(1, 2)).transpose(1, 2)
        u = F.silu(u)

        y = self.selective_scan(u)
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)

        return residual + y


class BenefitGatedSharedMambaHAR(nn.Module):
    """Benefit-gated shared Mamba model for adaptive HAR"""
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int,
        total_blocks: int,
        early_exit_blocks: int,
        dropout: float,
        gate_hidden_dim: int,
        conv_kernel: int,
    ):
        super().__init__()
        if not 1 <= early_exit_blocks < total_blocks:
            raise ValueError("early_exit_blocks must satisfy 1 <= early_exit_blocks < total_blocks")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.total_blocks = total_blocks
        self.early_exit_blocks = early_exit_blocks
        self.dropout_rate = dropout
        self.gate_hidden_dim = gate_hidden_dim
        self.conv_kernel = conv_kernel

        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList(
            [
                MiniMambaBlock(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    conv_kernel=conv_kernel,
                )
                for _ in range(total_blocks)
            ]
        )

        self.readout_norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        gate_input_dim = hidden_dim + num_classes + 6

        self.benefit_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )

    def pool_feature(self, z: torch.Tensor) -> torch.Tensor:
        z = self.readout_norm(z)
        return z.mean(dim=1)

    def classify_from_feature(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(h)

    def build_gate_input(
        self,
        x: torch.Tensor,
        h_early: torch.Tensor,
        early_logits: torch.Tensor,
    ) -> torch.Tensor:
        prob = F.softmax(early_logits, dim=1)
        top2 = torch.topk(prob, k=2, dim=1).values

        confidence = top2[:, 0:1]
        margin = top2[:, 0:1] - top2[:, 1:2]
        entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1, keepdim=True)
        entropy = entropy / math.log(self.num_classes)

        temporal_energy = (x[:, :, 1:] - x[:, :, :-1]).pow(2).mean(dim=(1, 2)).unsqueeze(1)
        signal_energy = x.pow(2).mean(dim=(1, 2)).unsqueeze(1)
        abs_mean = x.abs().mean(dim=(1, 2)).unsqueeze(1)

        return torch.cat(
            [
                h_early,
                early_logits,
                confidence,
                margin,
                entropy,
                temporal_energy,
                signal_energy,
                abs_mean,
            ],
            dim=1,
        )

    def forward_all(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.input_proj(x)
        z = z.transpose(1, 2)

        early_logits = None
        h_early = None

        for i, block in enumerate(self.blocks):
            z = block(z)

            if i + 1 == self.early_exit_blocks:
                h_early = self.pool_feature(z)
                early_logits = self.classify_from_feature(h_early)

        gate_input = self.build_gate_input(x, h_early, early_logits)
        gate_logit = self.benefit_gate(gate_input).squeeze(1)
        gate_prob = torch.sigmoid(gate_logit)

        h_final = self.pool_feature(z)
        final_logits = self.classify_from_feature(h_final)

        return early_logits, final_logits, gate_logit, gate_prob

    @torch.no_grad()
    def forward_dynamic(
        self,
        x: torch.Tensor,
        benefit_tau: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.input_proj(x)
        z = z.transpose(1, 2)

        for i in range(self.early_exit_blocks):
            z = self.blocks[i](z)

        h_early = self.pool_feature(z)
        early_logits = self.classify_from_feature(h_early)

        gate_input = self.build_gate_input(x, h_early, early_logits)
        gate_logit = self.benefit_gate(gate_input).squeeze(1)
        gate_prob = torch.sigmoid(gate_logit)

        full_mask = gate_prob >= benefit_tau

        output_logits = early_logits.clone()
        route = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if full_mask.any():
            z_full = z[full_mask]

            for i in range(self.early_exit_blocks, self.total_blocks):
                z_full = self.blocks[i](z_full)

            h_full = self.pool_feature(z_full)
            full_logits = self.classify_from_feature(h_full)

            output_logits[full_mask] = full_logits
            route[full_mask] = 1

        return output_logits, route, gate_prob
