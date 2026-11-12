# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: train.py/run.py에서 import. 미래GT와파일접근없음.
"""클래스별잔차와희귀Solwinner보호를분리한관측전용추론."""
from dataclasses import dataclass, replace
import numpy as np
import torch
from torch import nn
from adapters import Entry
from features import Features
from head import inputs

class Adaptive(nn.Module):
    def __init__(self, residual: bool):
        super().__init__()
        self.residual=residual
        self.layer=nn.Linear(94,26)
        if residual:
            nn.init.zeros_(self.layer.weight);nn.init.zeros_(self.layer.bias)

    def correction(self, x: torch.Tensor) -> torch.Tensor:
        return .75*self.layer(x).tanh()

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        return torch.logit(base.clamp(1e-5,1-1e-5))+self.correction(x) if self.residual else self.layer(x)

class LogAdaptive(Adaptive):
    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        return base.clamp(min=1e-6).log()+self.correction(x)

@dataclass(frozen=True, slots=True)
class Decoder:
    model: Adaptive
    support: np.ndarray
    guard: bool=False
    log_score: bool=False


def predict(entry: Entry, feature: Features, decoder: Decoder) -> Entry:
    x,indices=inputs(feature);base=feature.baseline[indices]
    rawbase=entry.relations[feature.rows[indices],feature.predicates[indices]]
    with torch.no_grad():
        if decoder.log_score:
            probabilities=rawbase*np.exp(decoder.model.correction(torch.from_numpy(x)).numpy())
        else:
            probabilities=decoder.model(torch.from_numpy(x),torch.from_numpy(base)).sigmoid().numpy().astype(entry.relations.dtype,copy=False)
    probabilities[:,decoder.support==0]=rawbase[:,decoder.support==0]
    if decoder.guard:
        for start,end in ((0,3),(3,9),(9,26)):
            winners=rawbase[:,start:end].argmax(1)+start
            preserve=decoder.support[winners]<128
            probabilities[preserve,start:end]=rawbase[preserve,start:end]
    scores=entry.relations.copy()
    scores[feature.rows[indices],feature.predicates[indices]]=probabilities
    return replace(entry,relations=scores)
