"""동일 관측 특징에서 손실·시간 구조·학습 메모리를 대조한다."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from typing_extensions import assert_never
import torch
from torch import nn
from torch.nn import functional as F

EncoderKind = Literal['gru', 'tcn', 'delta', 'delta_no_decay', 'delta_no_update', 'masked']


@dataclass(frozen=True, slots=True)
class Config:
    name: str
    idea: str
    encoder: EncoderKind = 'gru'
    residual: bool = False
    ccc_weight: float = 0.
    current_weight: float = 0.
    mask_weight: float = 0.
    group_balanced: bool = False
    epochs: int = 120
    seed: int = 20261111


CONFIGS = (
    Config('e01_gru_mse', 'E01'),
    Config('e02_gru_ccc005', 'E02', ccc_weight=.05),
    Config('e02_gru_ccc020', 'E02', ccc_weight=.2),
    Config('e03_residual_mse', 'E03', residual=True),
    Config('e03_residual_ccc', 'E03', residual=True, ccc_weight=.05),
    Config('e04_current_aux', 'E04', residual=True, ccc_weight=.05, current_weight=.5),
    Config('e05_group_balanced', 'E05', residual=True, ccc_weight=.05, group_balanced=True),
    Config('e06_multiscale_tcn', 'E06', encoder='tcn', residual=True, ccc_weight=.05),
    Config('e07_masked_temporal', 'E07', encoder='masked', residual=True, ccc_weight=.05, mask_weight=.1),
    Config('e07_no_mask_control', 'E07', encoder='masked', residual=True, ccc_weight=.05),
    Config('e08_gated_delta', 'E08', encoder='delta', residual=True, ccc_weight=.05),
    Config('e08_no_decay', 'E08', encoder='delta_no_decay', residual=True, ccc_weight=.05),
    Config('e08_no_delta_update', 'E08', encoder='delta_no_update', residual=True, ccc_weight=.05),
)


class DeltaEncoder(nn.Module):
    def __init__(self, kind: EncoderKind) -> None:
        super().__init__()
        self.kind = kind
        self.qkv = nn.Linear(33, 48)
        self.gates = nn.Linear(33, 2)
        self.output = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, 32), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        gates = self.gates(x).sigmoid()
        state = x.new_zeros((len(x), 16, 16))
        output = x.new_zeros((len(x), 16))
        for t in range(x.shape[1]):
            if self.kind != 'delta_no_decay':
                state = state * gates[:, t, 0, None, None]
            value = v[:, t]
            if self.kind != 'delta_no_update':
                value = value - torch.bmm(state, k[:, t, :, None]).squeeze(-1)
            state = state + gates[:, t, 1, None, None] * value[:, :, None] * k[:, t, None, :]
            output = torch.bmm(state, q[:, t, :, None]).squeeze(-1)
        return self.output(output)


class AffectModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        match config.encoder:
            case 'gru':
                self.encoder = nn.GRU(33, 32, batch_first=True)
            case 'tcn':
                self.convolutions = nn.ModuleList([nn.Conv1d(33, 32, k, padding=k // 2) for k in (3, 5, 7)])
                self.combine = nn.Linear(96, 32)
            case 'delta' | 'delta_no_decay' | 'delta_no_update':
                self.encoder = DeltaEncoder(config.encoder)
            case 'masked':
                self.project = nn.Linear(33, 32)
                self.position = nn.Parameter(torch.zeros(1, 15, 32))
                layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0., batch_first=True)
                self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
                self.reconstruct = nn.Linear(32, 33)
            case unreachable:
                assert_never(unreachable)
        self.head = nn.Linear(38, 8)
        self.current = nn.Linear(38, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.current.weight)
        nn.init.zeros_(self.current.bias)

    def forward(self, x: torch.Tensor, numeric: torch.Tensor, hold: torch.Tensor,
                masked: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reconstruction = x.new_zeros(())
        match self.config.encoder:
            case 'gru':
                encoded = self.encoder(x)[1][-1]
            case 'tcn':
                parts = [F.gelu(layer(x.transpose(1, 2))).mean(-1) for layer in self.convolutions]
                encoded = self.combine(torch.cat(parts, dim=-1)).tanh()
            case 'delta' | 'delta_no_decay' | 'delta_no_update':
                encoded = self.encoder(x)
            case 'masked':
                hidden = torch.rand(x.shape[:2], device=x.device) < .2 if masked else torch.zeros(x.shape[:2], device=x.device, dtype=torch.bool)
                hidden[:, -1] = False
                tokens = self.project(x.masked_fill(hidden[:, :, None], 0.)) + self.position
                sequence = self.encoder(tokens)
                encoded = sequence[:, -1]
                if masked:
                    errors = (self.reconstruct(sequence) - x).square().mean(-1)
                    reconstruction = (errors * hidden).sum() / hidden.sum().clamp_min(1)
            case unreachable:
                assert_never(unreachable)
        features = torch.cat([encoded, numeric], dim=-1)
        current = hold[:, 0] + self.current(features)
        prediction = self.head(features).reshape(-1, 4, 2)
        if self.config.residual:
            prediction = prediction + (current[:, None] if self.config.current_weight else hold)
        return prediction, current, reconstruction


def ccc_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p, y = prediction.flatten(1), target.flatten(1)
    covariance = ((p - p.mean(0)) * (y - y.mean(0))).mean(0)
    denominator = p.var(0, unbiased=False) + y.var(0, unbiased=False) + (p.mean(0) - y.mean(0)).square()
    return 1 - (2 * covariance / denominator.clamp_min(1e-8)).mean()
