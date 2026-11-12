# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
# How to run: imported by train.py and evaluate.py in the existing local environment.
"""메모리 gate·양성 가중·보정 상한을 분리해 조합한다."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import torch

spec = importlib.util.spec_from_file_location('combination_reference', Path(__file__).resolve().parents[1] / 'models.py')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
retrieve = base.retrieve


@dataclass(frozen=True, slots=True)
class Config(base.Config):
    balance_power: float = 0.
    cap: float = .75


CONFIGS = (
    Config('c01_positive_only', 'C01', class_balance=True, balance_power=.25),
    Config('c02_cap_only', 'C02', cap=3.),
    Config('c03_positive_cap', 'C03', class_balance=True, balance_power=.25, cap=3.),
    Config('c04_gate_positive_cap3', 'C04', memory=True, learned_mix=True, class_balance=True, balance_power=.25, cap=3.),
    Config('c05_gate_positive125_cap3', 'C05', memory=True, learned_mix=True, class_balance=True, balance_power=.125, cap=3.),
    Config('c06_gate_positive_cap15', 'C06', memory=True, learned_mix=True, class_balance=True, balance_power=.25, cap=1.5),
)
BLENDS = (('c07_combo4_baseline025', .25), ('c08_combo4_baseline050', .5))


class RelationHead(base.RelationHead):
    def forward(self, history: torch.Tensor, scene: torch.Tensor, delta: torch.Tensor,
                prior: torch.Tensor, memory: torch.Tensor, similarity: torch.Tensor) -> torch.Tensor:
        values = torch.cat([history.flatten(1), delta[:, None] / 30, prior], -1)
        correction = self.config.cap * self.head(values).tanh()
        prediction = (torch.logit(prior) + correction).sigmoid()
        if self.config.memory:
            entropy = -(prior * prior.log() + (1 - prior) * torch.log1p(-prior)).mean(1)
            features = torch.stack([delta / 30, entropy, similarity, (prior - memory).abs().mean(1)], -1)
            mix = self.gate(features).sigmoid()
            prediction = (1 - mix) * prediction + mix * memory
        return prediction.clamp(1e-6, 1 - 1e-6)


def blend_baseline(prediction: torch.Tensor, prior: torch.Tensor, fraction: float) -> torch.Tensor:
    """baseline 비중을 고정한 확률 혼합. 정답을 입력으로 받지 않는다."""
    assert 0 <= fraction <= 1
    return ((1 - fraction) * prediction + fraction * prior).clamp(1e-6, 1 - 1e-6)
