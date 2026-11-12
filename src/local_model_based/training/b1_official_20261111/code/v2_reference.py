"""구간 교정에 영향 없는 예측이 정확히 같을 때만 동결 v2 통계를 재사용한다."""
from functools import lru_cache
from pathlib import Path
import json
import torch
from typing import TypedDict
from provenance import digest
from sweep_data import HERE

REFERENCE = HERE.parent / 'config_sweep_v2'


class ReferenceSignature(TypedDict):
    freeze_sha256: str
    records: dict[str, str]


def affected(method: str) -> bool:
    return method.startswith('memory_') or '__' in method


def signature() -> ReferenceSignature:
    return {'freeze_sha256': digest(REFERENCE / 'protocol/test_freeze.json'),
            'records': {p.name: digest(p) for p in sorted((REFERENCE / 'results/test_records').glob('*.json'))}}


@lru_cache(maxsize=1)
def frozen_reference() -> ReferenceSignature:
    return json.loads((HERE / 'protocol/test_freeze.json').read_text())['v2_reference']


def matching_statistics(stem: str, outputs: dict) -> dict[str, dict]:
    frozen = frozen_reference()
    record_path = REFERENCE / f'results/test_records/{stem}.json'
    assert digest(record_path) == frozen['records'][record_path.name]
    record = json.loads(record_path.read_text())
    assert record['status'] == 'success' and record['freeze_sha256'] == frozen['freeze_sha256']
    saved = REFERENCE / f'results/test_predictions/{stem}.pt'
    assert digest(saved) == record['prediction_sha256']
    previous = torch.load(saved, weights_only=False)
    reused = {}
    for name, output in outputs.items():
        if affected(name):
            continue
        assert output.keys() == previous[name].keys(), name
        assert all(torch.equal(value, previous[name][key]) for key, value in output.items()), name
        reused[name] = record['methods'][name]
    assert len(reused) == 33
    return reused
