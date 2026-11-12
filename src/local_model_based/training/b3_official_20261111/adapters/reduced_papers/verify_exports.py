"""완료된 출력 전수 해시와 개발 출력 재현을 정답 없이 확인한다."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from adapter_io import anchor, install_guard, load_role, sha
from adapter_reduced import infer, model_module, role_signature
from observed_io import load_observed_role


def main() -> None:
    fold = int(sys.argv[1])
    folder = ROOT / 'adapters/reduced_papers' / f'fold{fold}'
    torch.set_num_threads(2)
    install_guard('predict', fold)
    selection = json.loads((folder / 'selection.json').read_text())
    for source, digest in selection['sources'].items():
        if source.startswith('/'):
            assert sha(Path(source)) == digest, source
    counts = {}
    for role in ('dev', 'eval'):
        batch, metadata, labels = load_role(fold, role, targets=False)
        assert labels is None
        receipt = json.loads((folder / f'{role}_receipt.json').read_text())
        assert receipt['role'] == role_signature(fold, role, metadata)
        for name, digest in receipt['outputs'].items():
            path = folder / role / f'{name}.npy'
            assert sha(path) == digest
            output = np.load(path, mmap_mode='r')
            assert output.shape == (len(metadata), 12, 2) and np.isfinite(output).all()
        counts[role] = {'rows': len(metadata), 'outputs': len(receipt['outputs'])}
    batch, metadata, labels = load_role(fold, 'dev', targets=False)
    observed = load_observed_role(fold, 'dev')
    local = anchor(fold, 'dev')
    arrays = (*observed, np.asarray([1 / row['fps'] for row in metadata], dtype=np.float32))
    module = model_module()
    differences = {}
    for checkpoint in sorted(folder.glob('*.pt')):
        name = checkpoint.stem
        receipt = json.loads(checkpoint.with_suffix('.receipt.json').read_text())
        assert sha(checkpoint) == receipt['checkpoint_sha256']
        fit_path = folder / 'fit_dev' / f'{name}.npy'
        assert sha(fit_path) == receipt['prediction_sha256']
        exported = np.load(folder / 'dev' / f'{name}.npy')
        fit_world = np.load(fit_path) + batch.origin[:, None]
        assert np.allclose(exported, fit_world, rtol=0, atol=1e-12)
        device = receipt['device']
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        assert saved['signature'] == receipt['signature']
        net = module.Core(module.Config(**saved['config'])).to(device)
        net.load_state_dict(saved['state_dict'])
        base = local if saved['kind'] == 'attention' else batch.cv
        actual = infer(net, tuple(value[:128] for value in arrays), base[:128], device)
        difference = float(np.max(np.abs(actual - exported[:128])))
        assert difference == 0, (name, difference)
        differences[name] = difference
    print(json.dumps({'fold': fold, 'counts': counts, 'first128_replay_maxdiff': differences,
                      'all_dev_matches_training_export': True, 'future_targets_loaded': False,
                      'API_calls': 0, 'verification_code_sha256': sha(Path(__file__))}))


if __name__ == '__main__':
    main()
