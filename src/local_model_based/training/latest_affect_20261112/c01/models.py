# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
# How to run: run.py에서 import한다.
"""작은 GRU 전문가와 OOF 기반 gate·변화폭 보정기."""
from dataclasses import dataclass
from typing import Final
import math
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class Config:
    name: str
    mtp: bool = False
    gate: bool = False
    correction: bool = False
    fixed_gate: float = 1.
    smooth: bool = False


CONFIGS: Final = (
    Config('B2-C01'), Config('B2-C02', mtp=True),
    Config('B2-C03', gate=True), Config('B2-C04', correction=True),
    Config('B2-C05', mtp=True, gate=True), Config('B2-C06', mtp=True, correction=True),
    Config('B2-C07', gate=True, correction=True), Config('B2-C08', mtp=True, gate=True, correction=True),
    Config('B2-C09', mtp=True, correction=True, fixed_gate=.5),
    Config('B2-C10', mtp=True, gate=True, correction=True, smooth=True),
)


class Expert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.GRU(33, 8, batch_first=True)
        self.project = nn.Linear(14, 16)
        self.horizon = nn.Parameter(torch.randn(1, 4, 16) * .02)
        self.output = nn.Linear(16, 2)
        self.value_embedding = nn.Linear(2, 16)
        self.auxiliary = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.LayerNorm(16))

    def forward(self, x: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        state = self.project(torch.cat((self.encoder(x)[1][-1], numeric), -1))[:, None]
        return state + self.horizon

    def predict(self, x: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        return self.output(self(x, numeric).tanh())

    def objective(self, inputs: tuple[torch.Tensor, torch.Tensor], target: torch.Tensor, mtp: bool) -> torch.Tensor:
        state = self(*inputs)
        loss = (self.output(state.tanh()) - target).square().mean()
        if mtp:
            next_state = self.auxiliary(torch.cat((state[:, :-1], self.value_embedding(target[:, :-1])), -1))
            loss = loss + .1 * (self.output(next_state) - target[:, 1:]).square().mean()
        return loss


def ccc_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p, y = prediction.flatten(1), target.flatten(1)
    covariance = ((p - p.mean(0)) * (y - y.mean(0))).mean(0)
    return 1 - (2 * covariance / (p.var(0, unbiased=False) + y.var(0, unbiased=False) + (p.mean(0) - y.mean(0)).square()).clamp_min(1e-8)).mean()


class Combiner(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.gate = nn.Sequential(nn.Linear(14, 16), nn.Tanh(), nn.Linear(16, 8))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(.1 / .9))
        self.a = nn.Parameter(torch.zeros(4, 2), requires_grad=config.correction)
        self.b = nn.Parameter(torch.zeros(4, 2), requires_grad=config.correction)
        for parameter in self.gate.parameters():
            parameter.requires_grad_(config.gate)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        expert, hold, observed = inputs
        gate = self.gate(torch.cat((observed, (expert - hold).abs().flatten(1)), -1)).sigmoid().reshape(-1, 4, 2) if self.config.gate else torch.full_like(expert, self.config.fixed_gate)
        a = 1 + .5 * self.a.tanh()
        b = .1 * self.b.tanh()
        return (hold + gate * (a * (expert - hold) + b)).clamp(-1, 1), gate
