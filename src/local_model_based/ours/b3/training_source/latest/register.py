# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scipy", "tokenizers"]
# ///
# How to run: CUDA_VISIBLE_DEVICES=0 python -B register.py
"""라벨 오차를 보기 전에 자산과 그룹 분리와 실행 설정을 동결한다."""
import json
import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import torch
from common import FOLDER, ROOT, OLD, ORIGINAL, SEEDS, EPOCHS, sha, save, freeze, setup, ids_sha


def main() -> None:
    setup()
    folds = {}
    for fold in (1, 2, 3):
        folder = FOLDER.parent / 'data' / f'fold{fold}'
        receipt = json.loads((folder / 'receipt.json').read_text())
        assert sha(folder / 'training.npz') == receipt['train_array_sha256']
        with np.load(folder / 'training.npz') as data:
            groups = data['groups']
            names, counts = np.unique(groups, return_counts=True)
            old_group = receipt['inner_validation_dataset']
            alternate = next(g for g in names if g != old_group)
            divisions = []
            for group in (old_group, alternate):
                train, validation = np.flatnonzero(groups != group), np.flatnonzero(groups == group)
                assert not set(groups[train]) & set(groups[validation])
                divisions.append({'train': len(train), 'validation': len(validation), 'train_sha': ids_sha(train), 'validation_sha': ids_sha(validation), 'disjoint_groups': True})
            folds[str(fold)] = {'groups': dict(zip(names.tolist(), counts.tolist())), 'validation_groups': [old_group, alternate],
                                'splits': divisions, 'input_sha256': sha(folder / 'training.npz')}
        for stage in ('inner', 'full'):
            parent = OLD / stage / f'fold{fold}' / '20261111/F/best.pt'
            record = json.loads((parent.parent / 'receipt.json').read_text())
            assert sha(parent) == record['checkpoint_sha256']
            checkpoint = torch.load(parent, map_location='cpu', weights_only=False)
            assert len(checkpoint['optimizer']['state']) == 10
    protected = {str(p): sha(p) for root in (OLD, ROOT / 'diagnosis_followup') for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    freeze(FOLDER / 'preservation_before.json', protected)
    freeze(FOLDER / 'registration.json', {'seeds': SEEDS, 'epoch_candidates': EPOCHS, 'folds': folds,
           'batch': 256, 'base_lr': .001, 'finetune_lr': .0001, 'max_additional_epochs': 10,
           'selection': 'within fold and seed: minimum worst held-out-group ADE/Kalman ratio; tie shorter epoch',
           'comparison_policy': 'same candidates, minimum original inner split ADE', 'device': 'GPU0', 'cpu_threads': 1,
           'sources': {str(p): sha(p) for p in [*FOLDER.glob('*.py'), OLD / 'models.py', OLD / 'engine.py', ROOT / 'followup_common.py', ORIGINAL / 'adapters/fast/model.py', ORIGINAL / 'code/collision_metric.py']}, 'decisions': sha(FOLDER / 'DECISIONS.md'),
           'python': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__, 'API_calls': 0,
           'downloads': 0, 'eval_count': 240890})
    save(FOLDER / 'processes_before.json', {'pid': os.getpid(), 'processes': subprocess.check_output(['ps', '-eo', 'pid,ppid,etimes,args'], text=True)})
    print(json.dumps({'registered': True, 'folds': folds}), flush=True)


if __name__ == '__main__':
    main()
