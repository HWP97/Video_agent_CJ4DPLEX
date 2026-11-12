# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: entry.py predict --sample predcls:GF7DV.mp4_0.5
"""관측 graph와 ImparTail prior에서 최신 선택 관계 head를 재계산한다."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from typing import Final

import numpy as np
import torch
from torch.nn import functional as F

ROOT: Final = Path(__file__).resolve().parents[2]
DATA: Final = ROOT / 'data/b1'
LATEST: Final = DATA / 'latest'
SELECTED: Final = {'sgcls': 'c04_gate_positive_cap3', 'predcls': 'c06_gate_positive_cap15'}
ACTIVE_FORWARD = [False]
sys.path.insert(0, str(ROOT / 'training/b1_official_20261111/code'))
from compact_graph import histories, observed_bank


def audit_forward(event: str, arguments: tuple) -> None:
    """프로세스에 한 번 설치하며 모델 실행 동안만 파일 접근을 차단한다."""
    if ACTIVE_FORWARD[0] and event == 'open':
        raise PermissionError('model_forward_file_access_forbidden')


sys.addaudithook(audit_forward)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=2)
def schedules(mode: str):
    cohort = 'filtered_small' if mode == 'sgcls' else 'unfiltered'
    return {r['case_id']: r for r in map(json.loads, (DATA / f'benchmark/protocol/{cohort}/cases.jsonl').read_text().splitlines())}


@lru_cache(maxsize=2)
def model_assets(mode: str):
    path = LATEST / f'{mode}_{SELECTED[mode]}.pt'
    receipt = json.loads(path.with_suffix('.json').read_text())
    assert digest(path) == receipt['checkpoint_sha256']
    state = torch.load(path, map_location='cpu', weights_only=False)['state']
    with np.load(LATEST / f'{mode}_memory.npz') as saved:
        memory = {key: saved[key] for key in saved.files}
    return state, memory


def predict(sample_id: str):
    """GT 없이 저장 관측과 동결 head만 사용하며 모델 구간의 파일 접근을 차단한다."""
    mode, case_id = sample_id.split(':', 1)
    assert mode in SELECTED
    stem = sample_id.replace(':', '__')
    receipt = json.loads((DATA / 'impartail/receipts' / f'{stem}.json').read_text())
    graph_path = DATA / 'impartail' / receipt['compact_path']
    prior_path = DATA / 'impartail' / receipt['prediction_path']
    assert digest(graph_path) == receipt['compact_sha256']
    assert digest(prior_path) == receipt['prediction_sha256']
    graph = torch.load(graph_path, map_location='cpu', weights_only=False)
    official = torch.load(prior_path, map_location='cpu', weights_only=False)
    schedule = schedules(mode)[case_id]
    future = [int(Path(name).stem) for name in schedule['future_frame_ids']]
    state, bank = model_assets(mode)
    ACTIVE_FORWARD[0] = True
    try:
        h, _ = histories(graph)
        n, count = len(future), len(h)
        history = h.repeat(n, 1, 1)
        delta = torch.tensor([(frame - graph['actual_relation_cutoff']) / 30 for frame in future]).repeat_interleave(count)
        prior = torch.cat((official['attention_distribution'].softmax(-1), official['spatial_distribution'], official['contacting_distribution']), -1).clamp(1e-5, 1-1e-5)
        bins = torch.bucketize(torch.arange(n), torch.linspace(0, n, 4).round().long()[1:-1], right=True).repeat_interleave(count)
        context = torch.cat((history[:, -1, :26], history[:, :, :26].mean(1), history[:, -1, 26:63], history[:, -1, 63:71]), -1)
        keys = F.normalize(torch.as_tensor(bank['context']), dim=-1)
        profiles = torch.as_tensor(bank['profile'])
        groups = list(bank['groups'])
        video = case_id.split('.mp4', 1)[0] + '.mp4'
        remembered, confidence = [], []
        for start in range(0, len(history), 256):
            similarities = F.normalize(context[start:start+256], dim=-1) @ keys.T
            excluded = torch.tensor([[video == group for group in groups]]).expand(len(similarities), -1)
            scores, indices = similarities.masked_fill(excluded, -torch.inf).topk(min(8, len(keys)), dim=1)
            assert torch.isfinite(scores).all()
            remembered.append(profiles[indices].mean(1).gather(2, bins[start:start+256, None, None].expand(-1,26,1)).squeeze(-1))
            confidence.append(scores[:, 0])
        memory, similarity = torch.cat(remembered), torch.cat(confidence)
        values = torch.cat((history.flatten(1), delta[:, None]/30, prior), -1)
        cap = 3.0 if mode == 'sgcls' else 1.5
        correction = cap * F.linear(values, state['head.weight'], state['head.bias']).tanh()
        corrected = (torch.logit(prior) + correction).sigmoid()
        entropy = -(prior*prior.log() + (1-prior)*torch.log1p(-prior)).mean(1)
        features = torch.stack((delta/30, entropy, similarity, (prior-memory).abs().mean(1)), -1)
        gate = F.linear(features, state['gate.weight'], state['gate.bias']).sigmoid()
        probability = ((1-gate)*corrected + gate*memory).clamp(1e-6,1-1e-6)
        geometry = observed_bank(graph, future)
    finally:
        ACTIVE_FORWARD[0] = False
    baseline = dict(geometry, **{key: official[key] for key in ('attention_distribution','spatial_distribution','contacting_distribution')})
    ours = dict(geometry, attention_distribution=probability[:,:3].clamp_min(1e-7).log(), spatial_distribution=probability[:,3:9], contacting_distribution=probability[:,9:])
    return baseline, ours, probability
