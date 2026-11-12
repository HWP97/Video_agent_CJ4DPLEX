"""Continuous masked restoration and causal auxiliary future prediction."""
from __future__ import annotations
from typing import Literal
from typing_extensions import assert_never
import torch
from torch import nn

Kind = Literal['parallel', 'masked', 'ar', 'mtp']

class Predictor(nn.Module):
    def __init__(self, inputs: int, kind: Kind) -> None:
        super().__init__()
        self.kind = kind
        self.encoder = nn.Sequential(nn.Linear(inputs, 64), nn.Tanh())
        match kind:
            case 'parallel' | 'masked':
                self.body = nn.Sequential(nn.Linear(100, 64), nn.Tanh(), nn.Linear(64, 24))
            case 'ar' | 'mtp':
                self.embedding = nn.Linear(2, 64)
                self.gru = nn.GRU(64, 64, batch_first=True)
                self.head = nn.Linear(64, 2)
                self.aux = nn.Sequential(nn.Linear(128, 64), nn.Tanh())
            case unreachable:
                assert_never(unreachable)

    def restore(self, x: torch.Tensor, state: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        clean = torch.where(visible[..., None], state, torch.zeros_like(state))
        return self.body(torch.cat((self.encoder(x), clean.flatten(1), visible.float()), 1)).reshape(-1, 12, 2)

    def teacher(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = torch.cat((torch.zeros_like(y[:, :1]), y[:, :-1]), 1)
        hidden, _ = self.gru(self.embedding(prefix), self.encoder(x)[None])
        main = self.head(hidden)
        auxiliary = self.head(self.aux(torch.cat((hidden[:, :-1], self.embedding(y[:, :-1])), -1)))
        return main, auxiliary

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        match self.kind:
            case 'parallel' | 'masked':
                state = x.new_zeros((len(x), 12, 2))
                visible = torch.zeros((len(x), 12), device=x.device, dtype=torch.bool)
                rounds = 3 if self.kind == 'masked' else 1
                for step in range(rounds):
                    prediction = self.restore(x, state, visible)
                    end = (step + 1) * (12 // rounds)
                    state[:, :end] = prediction[:, :end] if step == 0 else torch.cat((state[:, :step * 4], prediction[:, step * 4:end]), 1)
                    visible[:, :end] = True
                return state
            case 'ar' | 'mtp':
                hidden = self.encoder(x)[None]
                current = x.new_zeros((len(x), 1, 2))
                values = []
                for _ in range(12):
                    state, hidden = self.gru(self.embedding(current), hidden)
                    current = self.head(state)
                    values.append(current)
                return torch.cat(values, 1)
            case unreachable:
                assert_never(unreachable)

    def objective(self, x: torch.Tensor, y: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        match self.kind:
            case 'parallel' | 'masked':
                probability = x.new_ones((len(x), 1))
                if self.kind == 'masked':
                    probability = .1 + .9 * torch.rand_like(probability)
                    probability = torch.where(torch.rand_like(probability) < .25, torch.ones_like(probability), probability)
                masked = torch.rand((len(x), 12), device=x.device) < probability
                output = self.restore(x, y, ~masked)
                error = torch.linalg.vector_norm(output - y, dim=-1) * scales[:, None]
                return (error * masked / probability).mean()
            case 'ar' | 'mtp':
                main, auxiliary = self.teacher(x, y)
                loss = (torch.linalg.vector_norm(main - y, dim=-1) * scales[:, None]).mean()
                if self.kind == 'mtp':
                    loss = loss + .3 * (torch.linalg.vector_norm(auxiliary - y[:, 1:], dim=-1) * scales[:, None]).mean()
                return loss
            case unreachable:
                assert_never(unreachable)
