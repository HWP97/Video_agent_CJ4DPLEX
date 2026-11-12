from pathlib import Path
import sys
import json
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import sweep_predict as prediction
from sweep_models import Head, Denoiser
import sweep_data


def test_memory_decode_matches_training_profile_intervals(monkeypatch):
    torch.manual_seed(7)
    history = torch.zeros(1, 4, 71)
    monkeypatch.setattr(prediction, 'histories', lambda graph: (history, {}))
    monkeypatch.setattr(prediction, 'observed_bank', lambda graph, future: {})
    models = {name: Head(name) for name in prediction.PLAN['head_configs']}
    models.update({name: Denoiser() for name in prediction.PLAN['masked_patterns']})
    supports = {name: torch.ones(26) for name in models}
    profile = torch.tensor([.1, .5, .9]).expand(2, 26, 3).clone()
    loaded = (models, supports, {'context': torch.zeros(2, 97), 'profile': profile}, Head('linear_none'))
    for count in (1, 2, 4, 5, 7, 8, 12):
        original = {'attention_distribution': torch.zeros(count, 3), 'spatial_distribution': torch.zeros(count, 6), 'contacting_distribution': torch.zeros(count, 17)}
        result = prediction.predict({'actual_relation_cutoff': 0}, list(range(1, count + 1)), original, loaded)
        actual = 2 * result['memory_linear']['spatial_distribution'] - result['linear_none']['spatial_distribution']
        boundaries = torch.linspace(0, count, 4).round().long().tolist()
        for group in range(3):
            first, last = boundaries[group:group + 2]
            if first < last:
                assert torch.allclose(actual[first:last], torch.full_like(actual[first:last], [.1, .5, .9][group]), atol=1e-6), (count, group, actual)


def test_training_empty_bins_keep_existing_nearest_frame_fallback(tmp_path, monkeypatch):
    (tmp_path / 'protocol').mkdir()
    (tmp_path / 'assets/supervision').mkdir(parents=True)
    (tmp_path / 'protocol/train_manifest.json').write_text(json.dumps([{'mode': 'sgcls', 'status': 'ready', 'split': 'training', 'video': 'sample'}]))
    (tmp_path / 'assets/supervision/sgcls_sample.json').write_text('{}')
    monkeypatch.setattr(sweep_data, 'SOURCE', tmp_path)
    for count, expected in ((1, [0., 0., 0.]), (2, [0., 1., 1.])):
        original = {'attention_distribution': torch.zeros(count, 3), 'spatial_distribution': torch.zeros(count, 6), 'contacting_distribution': torch.zeros(count, 17)}
        target = {'predicted_history': torch.zeros(count, 4, 71), 'coverage': {'candidates': 1}, 'delta': torch.arange(count), 'future': torch.arange(count).float()[:, None].expand(count, 26)}
        monkeypatch.setattr(sweep_data.torch, 'load', lambda *args, **kwargs: {'graph': {}, 'official_prediction': original})
        monkeypatch.setattr(sweep_data, 'targets', lambda *args: target)
        packed, _ = sweep_data.pack('sgcls', 'development')
        assert torch.equal(packed['profile'], torch.tensor(expected).expand(1, 26, 3))
