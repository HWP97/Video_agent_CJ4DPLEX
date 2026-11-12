# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""Equal-capacity direct and restoration/residual relation dynamics."""
import torch
from torch import nn
from application_types import AdapterInputError


class RelationDynamics(nn.Module):
    def __init__(self, residual: bool, anchor: bool = False) -> None:
        super().__init__()
        self.residual = residual
        self.anchor = anchor
        self.subject = nn.Linear(30, 32)
        self.entity = nn.Linear(67, 32)
        self.position = nn.Parameter(torch.zeros(4, 64))
        self.temporal = nn.MultiheadAttention(64, 4, batch_first=True)
        self.drift = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64))
        self.diffusion = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64), nn.Softplus())
        self.current = nn.Linear(64, 26)
        self.future = nn.Linear(64, 26)

    def forward(self, history: torch.Tensor, delta: torch.Tensor, noise: torch.Tensor, base_logits: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        subject = torch.cat((history[..., :26], history[..., 63:67]), -1)
        entity = torch.cat((history[..., :63], history[..., 67:71]), -1)
        encoded = torch.cat((self.subject(subject).tanh(), self.entity(entity).tanh()), -1) + self.position
        last = encoded[:, -1:]
        aggregate, _ = self.temporal(last, encoded, encoded, need_weights=False)
        initial = (last + aggregate).squeeze(1)
        latent = initial
        step = delta.reshape(-1, 1) / len(noise)
        for increment in noise:
            latent = latent + self.drift(latent) * step + self.diffusion(latent) * increment * step.sqrt()
        current = self.current(initial)
        if self.anchor:
            if base_logits is None:
                raise AdapterInputError('Anchored head requires observed relation logits')
            current = current + base_logits
        future = self.future(latent)
        if self.residual:
            future = future - self.future(initial)
        horizon_zero = current if self.residual else current + self.future(initial)
        return current + future, horizon_zero


def focal(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid()
    pt = probability * target + (1 - probability) * (1 - target)
    return (-(.25 * target + .75 * (1 - target)) * (1 - pt).square() * pt.clamp_min(1e-7).log()).mean()
