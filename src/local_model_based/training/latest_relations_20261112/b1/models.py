"""빈도·영상 가중과 관측 구조·학습 메모리를 관계 예측에 적용한다."""
from __future__ import annotations
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

spec = importlib.util.spec_from_file_location('affect_memory_components', Path(__file__).resolve().parents[1] / 'b2/models.py')
components = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = components
spec.loader.exec_module(components)


@dataclass(frozen=True, slots=True)
class Config:
    name: str
    idea: str
    class_balance: bool = False
    group_balance: bool = False
    scene: bool = False
    delta: bool = False
    gated: bool = False
    memory: bool = False
    learned_mix: bool = False
    epochs: int = 20


CONFIGS = (
    Config('e01_linear_control', 'E01'),
    Config('e02_class_balance', 'E02', class_balance=True),
    Config('e03_video_balance', 'E03', group_balance=True),
    Config('e03_class_video_balance', 'E03', class_balance=True, group_balance=True),
    Config('e04_horizon_gate', 'E04', gated=True),
    Config('e05_scene_context', 'E05', scene=True),
    Config('e05_scene_balanced', 'E05', scene=True, class_balance=True),
    Config('e06_delta_memory', 'E06', delta=True),
    Config('e06_delta_balanced', 'E06', delta=True, class_balance=True),
    Config('e07_memory_half', 'E07', memory=True),
    Config('e08_memory_confidence_gate', 'E08', memory=True, learned_mix=True),
    Config('e08_memory_gate_balanced', 'E08', memory=True, learned_mix=True, class_balance=True),
)


class RelationHead(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        dimension = 311
        if config.scene:
            dimension += 71
        if config.delta:
            self.project = nn.Linear(71, 33)
            self.encoder = components.DeltaEncoder('delta')
            dimension = 32 + 27
        self.head = nn.Linear(dimension, 26)
        self.gate = nn.Linear(4, 26)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, history: torch.Tensor, scene: torch.Tensor, delta: torch.Tensor,
                base: torch.Tensor, memory: torch.Tensor, similarity: torch.Tensor) -> torch.Tensor:
        temporal = self.encoder(self.project(history)) if self.config.delta else history.flatten(1)
        values = [temporal, delta[:, None] / 30, base]
        if self.config.scene:
            values.append(scene[:, -1])
        correction = .75 * self.head(torch.cat(values, dim=-1)).tanh()
        entropy = -(base * base.log() + (1 - base) * torch.log1p(-base)).mean(1)
        gate_input = torch.stack([delta / 30, entropy, similarity, (base - memory).abs().mean(1)], dim=-1)
        gate = self.gate(gate_input).sigmoid()
        if self.config.gated:
            correction = correction * gate
        prediction = (torch.logit(base) + correction).sigmoid()
        if self.config.memory:
            mix = gate if self.config.learned_mix else .5
            prediction = (1 - mix) * prediction + mix * memory
        return prediction.clamp(1e-6, 1 - 1e-6)


def retrieve(history: torch.Tensor, bins: torch.Tensor, groups: list[str], bank,
             device: str = 'cuda') -> tuple[torch.Tensor, torch.Tensor]:
    context = torch.cat([history[:, -1, :26], history[:, :, :26].mean(1), history[:, -1, 26:63], history[:, -1, 63:71]], -1)
    keys = F.normalize(torch.as_tensor(bank['context'], device=device), dim=-1)
    profiles = torch.as_tensor(bank['profile'], device=device)
    bank_groups = list(bank['groups'])
    output, confidence = [], []
    for start in range(0, len(history), 256):
        similarities = F.normalize(context[start:start + 256], dim=-1) @ keys.T
        excluded = torch.tensor([[g == b for b in bank_groups] for g in groups[start:start + 256]], device=device)
        similarities = similarities.masked_fill(excluded, -torch.inf)
        scores, ids = similarities.topk(min(8, len(keys)), dim=1)
        assert torch.isfinite(scores).all()
        remembered = profiles[ids].mean(1)
        selected = remembered.gather(2, bins[start:start + 256, None, None].expand(-1, 26, 1)).squeeze(-1)
        output.append(selected)
        confidence.append(scores[:, 0])
    return torch.cat(output), torch.cat(confidence)
