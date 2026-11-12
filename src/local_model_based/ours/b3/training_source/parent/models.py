# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scipy", "tokenizers"]
# ///
# How to run: run.py에서 import한다.
"""원본 두 상태 반복 F에 관측 gate·개선 감독·불확실성 손실을 추가한다."""
from dataclasses import dataclass
from pathlib import Path
from typing import Final
import math
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'official_local_train_20261111/b3/adapters/fast'))
from model import Predictor


@dataclass(frozen=True, slots=True)
class Config:
    name: str
    gate: bool = False
    progress: bool = False
    uncertainty: bool = False
    central: bool = False
    fixed_gate: bool = False


CONFIGS: Final = (
    Config('B3-C01'), Config('B3-C02', gate=True), Config('B3-C03', progress=True),
    Config('B3-C04', uncertainty=True), Config('B3-C05', gate=True, progress=True),
    Config('B3-C06', gate=True, uncertainty=True), Config('B3-C07', progress=True, uncertainty=True),
    Config('B3-C08', gate=True, progress=True, uncertainty=True),
    Config('B3-C09', gate=True, progress=True, uncertainty=True, central=True),
    Config('B3-C10', gate=True, progress=True, uncertainty=True, fixed_gate=True),
)


def features(data):
    return torch.cat((data['ego'].flatten(1), (data['neighbors'] * data['mask'][:, :, None]).flatten(1), data['mask']), -1)


class Model(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.base = Predictor(125, 'two_state_deep')
        self.gate = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(.95 / .05))
        for parameter in self.gate.parameters():
            parameter.requires_grad_(config.gate and not config.fixed_gate)

    def forward(self, data):
        x = ((features(data) - data['center']) / data['spread']).clamp(-10, 10)
        proposals = [p.reshape(-1, 12, 2) for p in self.base.continuous(x)]
        previous = data['kalman']
        outputs, gates = [], []
        summary = torch.stack((data['ego'][:, -1, 2:4].norm(dim=-1), (data['ego'][:, -1, 2:4] - data['ego'][:, -2, 2:4]).norm(dim=-1), data['mask'].mean(1)), -1)
        for proposal in proposals:
            alpha = torch.ones_like(proposal[:, :, :1])
            if self.config.gate:
                horizon = torch.linspace(1 / 12, 1, 12, device=x.device)[None, :, None].expand(len(x), -1, -1)
                gate_input = torch.cat((summary[:, None].expand(-1, 12, -1), horizon, proposal - data['kalman'], proposal - previous), -1)
                alpha = torch.full_like(alpha, .5) if self.config.fixed_gate else self.gate(gate_input).sigmoid()
                value = previous + alpha * (proposal - previous)
            else:
                value = proposal
            outputs.append(value)
            gates.append(alpha)
            previous = value
        return outputs, gates


def noise_fit(data, ids: np.ndarray):
    """관측 첫7점의 마지막3점 직선으로 관측 뒤2점 잔차만 측정한다."""
    ego = data['ego'][ids, :, :2]
    history = ego[:, 4:7]
    velocity = (history[:, 2] - history[:, 0]) / 2
    future = history[:, -1, None] + torch.arange(1, 3, device=ego.device)[None, :, None] * velocity[:, None]
    errors = (ego[:, 7:9] - future) * data['scale'][ids, None, None]
    errors = errors / (torch.arange(1, 3, device=ego.device)[None, :, None] * data['dt'][ids, None, None])
    residual = errors.flatten(0, 1)
    cap = torch.quantile(residual.abs(), .95, dim=0)
    sigma = torch.minimum(residual.std(0, unbiased=False), cap)
    return sigma, cap


def collision_loss(outputs: list[torch.Tensor], data, specification: tuple) -> torch.Tensor:
    sigma, central = specification
    neighbors = data['neighbor_cv']
    offsets = torch.stack((sigma * 0, sigma * torch.tensor([1., 0.], device=sigma.device), sigma * torch.tensor([-1., 0.], device=sigma.device), sigma * torch.tensor([0., 1.], device=sigma.device), sigma * torch.tensor([0., -1.], device=sigma.device)))
    weights = torch.tensor([.4, .15, .15, .15, .15], device=sigma.device)
    if central:
        offsets, weights = offsets[:1], weights.new_ones(1)
    times = data['dt'][:, None] * torch.arange(1, 13, device=sigma.device)[None]
    hypotheses = neighbors[:, :, None] + offsets[None, None, :, None] * times[:, None, None, :, None] / data['scale'][:, None, None, None, None]
    neighbor_points = torch.cat((hypotheses, (hypotheses[:, :, :, :-1] + hypotheses[:, :, :, 1:]) / 2), 3)
    values = []
    for output in outputs:
        points = torch.cat((output, (output[:, :-1] + output[:, 1:]) / 2), 1)
        distance = (points[:, None, None] - neighbor_points).norm(dim=-1) * data['scale'][:, None, None, None]
        penalty = .05 * functional.softplus((.2 - distance) / .05)
        values.append((penalty * weights[None, None, :, None] * data['mask'][:, :, None, None]).sum() / (23 * data['mask'].sum()).clamp_min(1))
    return torch.stack(values).mean()


def objectives(outputs: list[torch.Tensor], data):
    errors = [(p - data['target']).norm(dim=-1).mean(1) * data['scale'] for p in outputs]
    prior = (data['kalman'] - data['target']).norm(dim=-1).mean(1) * data['scale']
    progress = []
    for error in errors:
        progress.append((error - prior.detach()).relu().mean())
        prior = error
    return torch.stack(errors).mean(), torch.stack(progress).mean()
