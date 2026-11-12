"""평가 정답을 읽지 않고 전체 어댑터 출력과 fold 출처를 검증한다."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from adapter_io import install_guard, sha


def verify(fold, readonly=False):
    install_guard('predict', fold)
    started = time.monotonic()
    plan = json.loads((ROOT / 'runplan.json').read_text())
    plan_sha = sha(ROOT / 'runplan.json')
    expected = {}
    order_hashes = {}
    for role, key in (('train', 'train'), ('dev', 'development'), ('eval', 'evaluation')):
        ids = []
        metadata = []
        for item in plan['groups'][plan['folds'][fold - 1][key]]:
            rows = json.loads((ROOT / 'data/indices' / (Path(item['file']).stem + '_index.json')).read_text())
            ids.extend(row['sample_id'] for row in rows)
            metadata.extend(rows)
        receipt = json.loads((ROOT / 'attention' / f'fold{fold}' / f'direct_unshared4_{role}.json').read_text())
        assert receipt['sample_ids'] == ids
        assert receipt['fold'] == fold and receipt['role'] == role
        assert receipt['runplan_sha256'] == plan_sha
        assert receipt['checkpoint_sha256'] == sha(ROOT / 'attention' / f'fold{fold}' / 'direct_unshared4.pt')
        expected[role] = len(ids)
        order_hashes[role] = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    checked = []
    for family, kinds in (('fast', ('continuous', 'raw_bpe', 'dct_bpe', 'two_state_deep', 'two_state_last', 'single_state')), ('masked', ('parallel', 'masked', 'ar', 'mtp'))):
        folder = ROOT / 'adapters' / family / f'fold{fold}'
        signature = json.loads((folder / 'input_signature.json').read_text())
        assert signature['runplan'] == plan_sha
        for path, digest in signature['source'].items():
            assert sha(ROOT / path) == digest, path
        for role_signature in signature['roles']:
            role = role_signature['role']
            for stem, digest in role_signature['data_receipts'].items():
                assert sha(ROOT / 'data/receipts' / (stem + '.json')) == digest
            assert role_signature['n'] == expected[role]
            assert role_signature['order_sha256'] == order_hashes[role]
            assert role_signature['anchor_sha256'] == sha(ROOT / 'attention' / f'fold{fold}' / f'direct_unshared4_{role}.npy')
        quantization = None
        if family == 'fast':
            codec = json.loads((folder / 'codec_receipt.json').read_text())
            assert codec['fingerprint'] == hashlib.sha256(json.dumps(signature, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            for path, digest in codec['artifacts'].items():
                assert sha(folder / path) == digest
            quantization = codec['scale']
        for arm in ('direct', 'local'):
            for kind in kinds:
                name = arm + '_' + kind
                receipt = json.loads((folder / (name + '.receipt.json')).read_text())
                assert receipt['signature'] == dict(signature, name=name, probe=False, quantization=quantization), 'method_or_configuration_mismatch'
                assert receipt['train_n'] == expected['train'] and receipt['dev_n'] == expected['dev']
                assert receipt['checkpoint_sha256'] == sha(folder / (name + '.pt'))
                assert receipt['prediction_sha256'] == sha(folder / 'dev' / (name + '.npy'))
                for role in ('dev', 'eval'):
                    path = folder / role / (name + '.npy')
                    values = np.load(path, mmap_mode='r')
                    assert values.shape == (expected[role], 12, 2)
                    assert all(np.isfinite(values[i:i + 8192]).all() for i in range(0, len(values), 8192))
                    checked.append({'family': family, 'method': name, 'role': role, 'shape': list(values.shape), 'sha256': sha(path)})
        for role in ('dev', 'eval'):
            inference = json.loads((folder / (role + '_inference_receipt.json')).read_text())
            assert inference['future_targets_loaded'] is False
            assert inference['role']['n'] == expected[role]
            assert inference['role']['order_sha256'] == order_hashes[role]
            for stem, digest in inference['role']['data_receipts'].items():
                assert sha(ROOT / 'data/receipts' / (stem + '.json')) == digest
            for row in checked:
                if row['family'] == family and row['role'] == role:
                    assert inference['outputs'][row['method']] == row['sha256']
    result = {'fold': fold, 'pass': True, 'counts': expected, 'arrays_checked': len(checked), 'arrays': checked, 'GT_read': False, 'api_calls': 0, 'seconds': time.monotonic() - started}
    path = ROOT / 'adapters/fast' / f'fold{fold}_verification.json'
    if not readonly:
        path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'arrays'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fold', type=int, choices=[1, 2, 3])
    parser.add_argument('--readonly', action='store_true')
    args = parser.parse_args()
    verify(args.fold, args.readonly)
