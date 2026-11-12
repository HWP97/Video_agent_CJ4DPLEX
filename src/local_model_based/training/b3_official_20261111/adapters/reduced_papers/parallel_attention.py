"""원 fit 함수로 fold1의 마지막 attention MOIF 네 설정을 별도 학습한다."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from adapter_io import anchor, install_guard, load_role, sha
from adapter_reduced import fit, model_module, save
from observed_io import load_observed_role


def main() -> None:
    probe = sys.argv[1:] == ['probe']
    assert probe or not sys.argv[1:]
    assert sha(ROOT / 'code/adapter_reduced.py') == '4de492e14114b7ac4de8116492a2a5ace274d20db8869fe7fafd368b9f1ae83c'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    install_guard('train', 1)
    folder = ROOT / 'adapters/reduced_papers' / ('parallel_probe' if probe else 'parallel_fold1_gpu1')
    (folder / 'fit_dev').mkdir(parents=True, exist_ok=True)
    packed = {}
    for role in ('train', 'dev'):
        batch, metadata, truth = load_role(1, role, targets=True)
        observed = load_observed_role(1, role)
        local = anchor(1, role)
        if probe:
            ids = np.arange(64)
            batch = batch.take(ids)
            truth = truth[ids]
            local = local[ids]
            observed = tuple(value[ids] for value in observed)
            metadata = [metadata[i] for i in ids]
        arrays = (*observed, np.asarray([1 / row['fps'] for row in metadata], dtype=np.float32))
        packed[role] = (arrays, local - batch.origin[:, None], truth - batch.origin[:, None], metadata)
    module = model_module()
    for config in module.CONFIGS:
        if config.family != 'moif':
            continue
        started = time.time()
        name = fit(config, 'attention', 1, probe, packed['train'], packed['dev'], folder, 'cuda')
        save(folder / f'{name}.producer.json', {
            'runner_sha256': sha(Path(__file__)), 'original_fit_source_sha256': sha(ROOT / 'code/adapter_reduced.py'),
            'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'device': 'cuda', 'pid': os.getpid(),
            'start_unix': started, 'end_unix': time.time(), 'probe_only': probe,
            'same_original_fit_function': True, 'publication_pending': True, 'API_calls': 0,
        })


if __name__ == '__main__':
    main()
