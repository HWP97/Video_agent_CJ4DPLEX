from pathlib import Path
import sys
import json
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import v2_reference as reference
from provenance import digest


def test_reuse_requires_identical_raw_and_frozen_statistics(tmp_path, monkeypatch):
    (tmp_path / 'results/test_records').mkdir(parents=True)
    (tmp_path / 'results/test_predictions').mkdir()
    outputs = {f'head_{i}': {'score': torch.tensor([float(i)])} for i in range(33)}
    prediction = tmp_path / 'results/test_predictions/sample.pt'
    torch.save(outputs, prediction)
    record = tmp_path / 'results/test_records/sample.json'
    record.write_text(json.dumps({'status': 'success', 'freeze_sha256': 'frozen', 'prediction_sha256': digest(prediction), 'methods': {name: {'R': i} for i, name in enumerate(outputs)}}))
    frozen = {'freeze_sha256': 'frozen', 'records': {'sample.json': digest(record)}}
    monkeypatch.setattr(reference, 'REFERENCE', tmp_path)
    monkeypatch.setattr(reference, 'frozen_reference', lambda: frozen)
    assert len(reference.matching_statistics('sample', outputs)) == 33
    outputs['head_0']['score'][0] = 100
    with pytest.raises(AssertionError):
        reference.matching_statistics('sample', outputs)
    outputs['head_0']['score'][0] = 0
    record.write_text(record.read_text().replace('"R": 0', '"R": 999'))
    with pytest.raises(AssertionError):
        reference.matching_statistics('sample', outputs)
