"""영상 단위 내부 분할과 기존 관측 전용 관계 특징을 저장한다."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parents[1] / 'official_local_train_20261111/b1/config_sweep_v3'
SOURCE = Path('/mnt/RG2/paper_experiments_20261109/combinations')
sys.path.insert(0, str(ORIGINAL / 'code'))
from supervision import targets
from sweep_data import probabilities, context
from compact_graph import histories


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    torch.set_num_threads(2)
    destination = ROOT / 'data'
    destination.mkdir(exist_ok=False)
    manifest = json.loads((SOURCE / 'protocol/train_manifest.json').read_text())
    training_videos = sorted({r['video'] for r in manifest if r['split'] == 'training'},
        key=lambda name: hashlib.sha256(('20261111:' + name).encode()).hexdigest())
    validation = set(training_videos[:80])
    records = {}
    started = time.time()
    for mode in ('sgcls', 'predcls'):
        rows = [r for r in manifest if r['mode'] == mode and r['status'] == 'ready']
        arrays = {key: [] for key in ('history', 'scene', 'delta', 'base', 'target', 'groups', 'split', 'bin')}
        memory = {key: [] for key in ('context', 'profile', 'groups', 'split')}
        for row in rows:
            saved = torch.load(SOURCE / 'assets/compact_train' / f'{mode}_{row["video"]}.pt', weights_only=False)
            supervision = json.loads((SOURCE / 'assets/supervision' / f'{mode}_{row["video"]}.json').read_text())
            data = targets(saved['graph'], supervision)
            h, coverage = histories(saved['graph'])
            candidates = coverage['candidates']
            n = len(data['future']) // candidates
            label = data['future'].reshape(n, candidates, 26)
            split = 'outer_dev' if row['split'] == 'development' else 'inner_validation' if row['video'] in validation else 'inner_train'
            arrays['history'].append(data['predicted_history'].numpy())
            arrays['scene'].append(h.mean(0, keepdim=True).expand(n * candidates, -1, -1).numpy())
            arrays['delta'].append(data['delta'].numpy())
            arrays['base'].append(probabilities(saved['official_prediction']).numpy())
            arrays['target'].append(data['future'].numpy())
            arrays['groups'].append(np.repeat(row['video'], n * candidates))
            arrays['split'].append(np.repeat(split, n * candidates))
            bins = torch.linspace(0, n, 4).round().long()
            frame_bins = torch.bucketize(torch.arange(n), bins[1:-1], right=True)
            arrays['bin'].append(frame_bins.repeat_interleave(candidates).numpy())
            profile = torch.stack([label[min(int(bins[i]), n - 1):max(int(bins[i + 1]), min(int(bins[i]), n - 1) + 1)].mean(0) for i in range(3)], -1)
            memory['context'].append(context(h).numpy())
            memory['profile'].append(profile.numpy())
            memory['groups'].append(np.repeat(row['video'], candidates))
            memory['split'].append(np.repeat(split, candidates))
        joined = {key: np.concatenate(values) for key, values in arrays.items()}
        assert not set(joined['groups'][joined['split'] == 'inner_train']) & set(joined['groups'][joined['split'] == 'inner_validation'])
        np.savez(destination / f'{mode}.npz', **joined)
        np.savez(destination / f'{mode}_memory.npz', **{key: np.concatenate(values) for key, values in memory.items()})
        records[mode] = {'rows': len(joined['base']), 'videos': len(rows),
            'split_rows': {s: int(np.sum(joined['split'] == s)) for s in np.unique(joined['split'])}}
    receipt = {'records': records, 'manifest_sha256': sha(SOURCE / 'protocol/train_manifest.json'),
        'source_sha256': sha(Path(__file__)), 'upstream_sources': {name: sha(ORIGINAL / 'code' / name) for name in ('supervision.py', 'compact_graph.py', 'sweep_data.py')},
        'outputs': {p.name: sha(p) for p in destination.glob('*.npz')},
        'test_labels_read': False, 'outer_dev_used_only_after_selection_for_full_refit': True,
        'seconds': time.time() - started, 'API_calls': 0}
    (destination / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
