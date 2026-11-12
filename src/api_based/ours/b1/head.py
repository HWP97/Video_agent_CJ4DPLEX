# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: train_head.py/run.py가 import. GT와 파일 접근 없는 inference.
"""객체 조건부 family 술어 분류 head."""
from dataclasses import replace
import numpy as np
import torch
from adapters import Entry
from features import Features

def inputs(feature: Features) -> tuple[np.ndarray, np.ndarray]:
    indices = np.stack([np.flatnonzero(feature.rows % feature.candidates == i) for i in range(feature.candidates)])
    assert np.all(feature.predicates[indices] == np.arange(26))
    first = feature.values[indices[:,0]]
    values = np.concatenate((first[:,7:44], first[:,70:96], feature.baseline[indices], first[:,3:7], first[:,96:97]),axis=1)
    assert values.shape[1] == 94
    return values, indices

def predict(entry: Entry, feature: Features, model: torch.nn.Linear) -> Entry:
    values, indices = inputs(feature)
    with torch.no_grad():
        probs = model(torch.from_numpy(values)).sigmoid().numpy()
    scores = entry.relations.copy()
    scores[feature.rows[indices],feature.predicates[indices]] = probs
    return replace(entry,relations=scores)
