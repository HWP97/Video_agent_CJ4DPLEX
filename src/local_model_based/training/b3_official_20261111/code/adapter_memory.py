"""기존 train-only memory와 관측 prefix 보정의 순수 수치 연산."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
import hashlib
import json
import os
import sys
import time


def prefix_delta(positions: np.ndarray, dt: np.ndarray, proximal: float = 1.) -> np.ndarray:
    assert positions.ndim == 3 and positions.shape[1:] == (9, 2)
    assert dt.shape == (len(positions),) and (dt > 0).all()
    assert np.isfinite(positions).all() and np.isfinite(dt).all()
    velocity = (positions[:, 5] - positions[:, 3]) / (2 * dt[:, None])
    elapsed = dt[:, None] * np.arange(1, 4)[None]
    error = positions[:, 6:9] - positions[:, 5, None] - elapsed[:, :, None] * velocity[:, None]
    return np.sum(elapsed[:, :, None] * error, axis=1) / (np.sum(elapsed**2, axis=1)[:, None] + proximal)


def ridge(x: np.ndarray, y: np.ndarray, regularizer: float) -> np.ndarray:
    return np.linalg.solve(x.T @ x + regularizer * len(x) * np.eye(x.shape[1]), x.T @ y.reshape(len(y), -1))


class MemoryIndex:
    def __init__(self, keys: np.ndarray) -> None:
        self.keys = np.asarray(keys, dtype=np.float64).reshape(len(keys), -1)
        assert len(self.keys) >= 20 and np.isfinite(self.keys).all()
        self.tree = cKDTree(self.keys)

    def neighbors(self, query: np.ndarray, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query, dtype=np.float64).reshape(len(query), -1)
        assert q.shape[1] == self.keys.shape[1] and np.isfinite(q).all()
        count = min(k + 1, len(self.keys))
        all_ids, all_distances = [], []
        for start in range(0, len(q), 128):
            block = q[start:start + 128]
            distances, indices = self.tree.query(block, k=count, eps=0, workers=2)
            for row, ds, ids in zip(block, distances, indices):
                if count > k and ds[k - 1] == ds[k]:
                    ids = np.asarray(self.tree.query_ball_point(row, np.nextafter(ds[k - 1], np.inf)), dtype=np.int64)
                    ds = np.linalg.norm(self.keys[ids] - row, axis=1)
                selected = np.lexsort((ids, ds))[:k]
                ids = ids[selected]
                squared = np.maximum((row**2).sum() + (self.keys[ids]**2).sum(1) - 2 * (self.keys[ids] @ row), 0)
                all_ids.append(ids)
                all_distances.append(squared)
        return np.stack(all_ids), np.stack(all_distances)


def retrieved(values: np.ndarray, ids: np.ndarray, distances: np.ndarray, k: int, weighted: bool) -> np.ndarray:
    weights = 1 / np.maximum(distances[:, :k], 1e-8) if weighted else np.ones((len(ids), k))
    weights /= weights.sum(1, keepdims=True)
    return np.sum(values[ids[:, :k]] * weights[:, :, None, None], axis=1)


def normalize(batch, coordinates: np.ndarray) -> np.ndarray:
    return np.matmul(coordinates - batch.origin[:, None], batch.rotation) / batch.scale[:, None, None]


def decode_shift(batch, offset: np.ndarray) -> np.ndarray:
    return np.matmul(offset, batch.rotation.transpose(0, 2, 1)) * batch.scale[:, None, None]


def fit_memory(batch, labels: np.ndarray, local: np.ndarray, dt: np.ndarray) -> dict[str, np.ndarray]:
    positions = batch.ego[:, :, :2]
    delta = prefix_delta(positions, dt)
    velocity = (positions[:, 5] - positions[:, 3]) / (2 * dt[:, None])
    x = np.column_stack((positions.reshape(len(positions), -1), np.ones(len(positions))))
    model = {'keys': positions.copy(), 'direct': normalize(batch, labels)}
    for name, baseline in (('kalman', batch.kalman), ('local', local)):
        model[name] = normalize(batch, labels) - normalize(batch, baseline)
        for regularizer in (.00001, .001, .1, 1., 10.):
            model[f'{name}_adapt_{regularizer}'] = ridge(delta, model[name], regularizer)
        for regularizer in (.1, 1., 10.):
            model[f'{name}_static_{regularizer}'] = ridge(x, model[name], regularizer)
    for regularizer in (.001, .01, .1):
        model[f'direct_adapt_{regularizer}'] = ridge(velocity + delta, model['direct'], regularizer)
        model[f'direct_ordinary_{regularizer}'] = ridge(velocity, model['direct'], regularizer)
    model['shuffled'] = model['local'][np.random.default_rng(20261110).permutation(len(positions))]
    return model


def predict_memory(batch, local: np.ndarray, dt: np.ndarray, model: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    positions = batch.ego[:, :, :2]
    delta = prefix_delta(positions, dt)
    velocity = (positions[:, 5] - positions[:, 3]) / (2 * dt[:, None])
    x = np.column_stack((positions.reshape(len(positions), -1), np.ones(len(positions))))
    ids, distances = MemoryIndex(model['keys']).neighbors(positions)
    anchors = {'kalman': batch.kalman, 'local': local}
    outputs = {name: value.copy() for name, value in anchors.items()}
    for k in (1, 5, 20):
        for weighted in (False, True):
            kind = 'weighted' if weighted else 'uniform'
            outputs[f'memory_direct_{kind}_k{k}'] = batch.origin[:, None] + decode_shift(batch, retrieved(model['direct'], ids, distances, k, weighted))
        for name, baseline in anchors.items():
            outputs[f'{name}_memory_k{k}'] = baseline + decode_shift(batch, retrieved(model[name], ids, distances, k, True))
    outputs['local_memory_shuffled_k5'] = local + decode_shift(batch, retrieved(model['shuffled'], ids, distances, 5, True))
    for name, baseline in anchors.items():
        for regularizer in (.00001, .001, .1, 1., 10.):
            shift = (delta @ model[f'{name}_adapt_{regularizer}']).reshape(-1, 12, 2)
            outputs[f'{name}_adapt_r{regularizer}'] = baseline + decode_shift(batch, shift)
        for regularizer in (.1, 1., 10.):
            shift = (x @ model[f'{name}_static_{regularizer}']).reshape(-1, 12, 2)
            outputs[f'{name}_static_r{regularizer}'] = baseline + decode_shift(batch, shift)
        outputs[f'{name}_identity_update'] = baseline + decode_shift(batch, delta[:, None] * (dt[:, None] * np.arange(1, 13)[None])[:, :, None])
        outputs[f'{name}_update_disabled'] = baseline.copy()
    for regularizer in (.001, .01, .1):
        for fit, inputs, suffix in (('adapt', velocity + delta, 'adapted'), ('adapt', velocity, 'update_disabled'), ('ordinary', velocity + delta, 'ordinaryfit_updated'), ('ordinary', velocity, 'ordinaryfit_unadapted')):
            relative = (inputs @ model[f'direct_{fit}_{regularizer}']).reshape(-1, 12, 2)
            outputs[f'direct_prefix_{suffix}_r{regularizer}'] = batch.origin[:, None] + decode_shift(batch, relative)
    assert all(value.shape == (len(positions), 12, 2) and np.isfinite(value).all() for value in outputs.values())
    return outputs


def source_signature(fold: int, role: str) -> dict[str, int | str | dict[str, str]]:
    from adapter_io import ROOT, sha
    plan = json.loads((ROOT / 'runplan.json').read_text())
    group = plan['folds'][fold - 1][{'train': 'train', 'dev': 'development', 'eval': 'evaluation'}[role]]
    files = [Path(__file__), ROOT / 'code/adapter_io.py', ROOT / 'code/features.py', ROOT / 'runplan.json', ROOT / 'attention' / f'fold{fold}' / f'direct_unshared4_{role}.json',
             Path('/mnt/RG2/llm_transfer_expansion_20261110/b3/experiment.py'), Path('/mnt/RG2/llm_transfer_expansion_20261110/b3/wave2.py')]
    files.extend(ROOT / 'data/receipts' / (Path(row['file']).stem + '.json') for row in plan['groups'][group])
    return {'fold': fold, 'role': role, 'group': group, 'files': {str(path): sha(path) for path in files}}


def role_metadata(metadata: list[dict]) -> dict[str, int | str]:
    return {'n': len(metadata), 'order_sha256': hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}


def main() -> None:
    from adapter_io import ROOT, anchor, install_guard, load_role, sha
    assert all(int(os.environ.get(key,'2'))<=2 for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'CPU thread limit must be at most 2'
    action, fold_text = sys.argv[1:3]
    fold = int(fold_text)
    assert action in ('train', 'predict-dev', 'predict-eval', 'pilot') and fold in (1, 2, 3)
    role = 'train' if action in ('train', 'pilot') else action.removeprefix('predict-')
    install_guard('train' if action in ('train', 'pilot') else 'predict', fold)
    folder = ROOT / 'adapters/memory' / f'fold{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    signature = source_signature(fold, 'train')
    checkpoint = folder / 'memory.npz'
    receipt_path = folder / 'training.json'
    if action in ('train', 'pilot'):
        batch, metadata, labels = load_role(fold, 'train', targets=True)
        local = anchor(fold, 'train')
        if action == 'pilot':
            ids = np.arange(min(64, len(metadata)))
            batch, labels, local = batch.take(ids), labels[ids], local[ids]
            metadata = [metadata[i] for i in ids]
        dt = np.array([1 / row['fps'] for row in metadata])
        if action == 'train' and receipt_path.exists():
            old = json.loads(receipt_path.read_text())
            assert old['signature'] == signature and old['checkpoint_sha256'] == sha(checkpoint)
            print(json.dumps({'status': 'verified_existing', 'n': old['n']}))
            return
        model = fit_memory(batch, labels, local, dt)
        if action == 'pilot':
            predicted = predict_memory(batch, local, dt, model)
            result = {'status': 'pilot_only', 'n': len(metadata), 'methods': len(predicted), 'seconds': time.monotonic() - started, 'API_calls': 0, 'full_training_complete': False}
            (folder / 'pilot.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result))
            return
        assert not checkpoint.exists()
        np.savez_compressed(checkpoint, **model)
        result = {'signature': signature, 'checkpoint_sha256': sha(checkpoint), 'n': len(metadata), 'role': role_metadata(metadata), 'sample_id_order_sha256': hashlib.sha256(json.dumps([row['sample_id'] for row in metadata]).encode()).hexdigest(),
                  'seed': 20261110, 'fit': '원 ridge 및 train-only memory grid', 'seconds': time.monotonic() - started, 'peak_vram_bytes': 0, 'API_calls': 0}
        receipt_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        trained = json.loads(receipt_path.read_text())
        assert trained['signature'] == signature and trained['checkpoint_sha256'] == sha(checkpoint)
        with np.load(checkpoint) as saved:
            model = {name: saved[name] for name in saved.files}
        batch, metadata, labels = load_role(fold, role, targets=False)
        assert labels is None
        values = predict_memory(batch, anchor(fold, role), np.array([1 / row['fps'] for row in metadata]), model)
        destination = folder / role
        destination.mkdir(exist_ok=False)
        outputs = {}
        for method, prediction in values.items():
            path = destination / f'{method}.npy'
            np.save(path, prediction)
            outputs[method] = sha(path)
        result = {'signature': source_signature(fold, role), 'checkpoint_sha256': sha(checkpoint), 'n': len(metadata),
                  'role': role_metadata(metadata), 'sample_id_order_sha256': hashlib.sha256(json.dumps([row['sample_id'] for row in metadata]).encode()).hexdigest(), 'outputs': outputs,
                  'seconds': time.monotonic() - started, 'peak_vram_bytes': 0, 'API_calls': 0, 'fallback_count': 0, 'failures': [],
                  'selection': 'selected_memory/adapt는 controller의 원 개발 ADE/FDE/관측 이웃 충돌 gate에서 결정'}
        (folder / f'{role}_receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key not in ('signature', 'outputs')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
