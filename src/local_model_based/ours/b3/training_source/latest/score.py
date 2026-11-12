# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scipy", "tokenizers"]
# ///
# How to run: python -B score.py
"""원본 충돌 scorer로 새 예측을 채점하고 완료 참조 통계를 재사용한다."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import sys
import numpy as np
from common import FOLDER, OLD, ORIGINAL, sha, save, freeze
sys.path.insert(0, str(ORIGINAL / 'code'))
from collision_metric import collision_segments


def score_file(task: tuple[int, str, int, str]) -> str:
    """동일 index 순서로 한 원본 파일의 모든 새 조건을 채점한다."""
    fold, stem, offset, request_hash = task
    request = json.loads((FOLDER / f'score_request_fold{fold}.json').read_text())
    assert sha(FOLDER / f'score_request_fold{fold}.json') == request_hash
    destination = FOLDER / 'scores' / f'fold{fold}'
    output = destination / (stem + '.npz')
    receipt_path = output.with_suffix('.json')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['request_sha256'] == request_hash and sha(output) == receipt['sha256']
        return str(output)
    index = ORIGINAL / 'data/indices' / f'{stem}_index.json'
    labels = ORIGINAL / 'data/targets' / f'{stem}_labels.npz'
    raw_path = ORIGINAL / 'evaluation_only' / f'{stem}.npz'
    rows = json.loads(index.read_text())
    n = len(rows)
    data_receipt = json.loads((ORIGINAL / 'data/receipts' / f'{stem}.json').read_text())
    for path in (index, labels):
        assert sha(path) == data_receipt['outputs'][str(path.relative_to(ORIGINAL))]
    auxiliary = json.loads((ORIGINAL / 'auxiliary_receipts' / f'{stem}.json').read_text())
    assert sha(raw_path) == auxiliary['outputs'][str(raw_path.relative_to(ORIGINAL))]
    names = list(request['predictions'])
    reuse = {'baseline_attention': 'baseline_attention', 'previous_F': 'previous_F', 'B3-D02': 'B3-C01'}
    new_names = [name for name in names if name not in reuse]
    values = np.stack([np.load(request['predictions'][name], mmap_mode='r')[offset:offset + n] for name in new_names])
    assert values.shape == (len(new_names), n, 12, 2) and np.isfinite(values).all()
    with np.load(labels) as y:
        distance = np.linalg.norm(values - y['xy'][None], axis=-1)
    ade, fde = distance.mean(-1), distance[:, :, -1]
    collision = np.zeros((len(new_names), n, 2), bool)
    with np.load(raw_path) as raw:
        tracks = raw['tracks']
        scenes = {int(r[0]): int(r[1]) for r in raw['scenes']}
    tracks = tracks[np.argsort(tracks[:, 0], kind='stable')]
    frames, first, counts = np.unique(tracks[:, 0], return_index=True, return_counts=True)
    by_frame = {int(f): tracks[start:start + count] for f, start, count in zip(frames, first, counts)}
    for i, row in enumerate(rows):
        primary = scenes[int(row['sample_id'].rsplit('_', 1)[1])]
        visible = np.unique(np.concatenate([by_frame[f][:, 1] for f in row['observed_frames']]))
        pieces = []
        for step, frame in enumerate(row['future_frames']):
            neighbors = by_frame[frame]
            neighbors = neighbors[neighbors[:, 1] != primary]
            pieces.append(np.column_stack([np.full(len(neighbors), step), neighbors[:, 1:]]))
        future = np.concatenate(pieces)
        future = future[np.argsort(future[:, 1], kind='stable')]
        paired = future[:-1, 1] == future[1:, 1]
        left, right = future[:-1][paired], future[1:][paired]
        all_hit, obs_hit = collision_segments(values[:, i], left[:, 0].astype(int), right[:, 0].astype(int), left[:, 2:], right[:, 2:], np.isin(left[:, 1], visible))
        collision[:, i, 0], collision[:, i, 1] = all_hit, obs_hit
    evaluation_paths = json.loads((OLD / 'evaluation_paths.json').read_text())
    old_folder = Path(evaluation_paths[str(fold)])
    old_path = old_folder / output.name
    old_receipt = json.loads((old_folder / 'receipt.json').read_text())
    assert sha(old_path) == old_receipt['outputs'][old_path.name]
    final_ade, final_fde, final_collision = [], [], []
    with np.load(old_path) as original:
        assert original['sample_ids'].tolist() == [r['sample_id'] for r in rows]
        for name in names:
            if name in reuse:
                j = original['methods'].tolist().index(reuse[name])
                final_ade.append(original['ADE'][j])
                final_fde.append(original['FDE'][j])
                final_collision.append(original['collision'][j])
            else:
                j = new_names.index(name)
                final_ade.append(ade[j])
                final_fde.append(fde[j])
                final_collision.append(collision[j])
    np.savez(output, ADE=np.stack(final_ade), FDE=np.stack(final_fde), collision=np.stack(final_collision),
             methods=np.array(names), sample_ids=np.array([r['sample_id'] for r in rows]))
    save(receipt_path, {'request_sha256': request_hash, 'sha256': sha(output), 'n': n,
         'reused_reference': str(old_path), 'reused_reference_sha256': sha(old_path),
         'labels_sha256': sha(labels), 'raw_sha256': sha(raw_path), 'index_sha256': sha(index),
         'metric_sha256': sha(ORIGINAL / 'code/collision_metric.py')})
    print(json.dumps({'fold': fold, 'file': stem, 'n': n}), flush=True)
    return str(output)


def main() -> None:
    freeze(FOLDER / 'score_registration.json', {'source': sha(Path(__file__)), 'metric': sha(ORIGINAL / 'code/collision_metric.py'),
           'workers': 6, 'reuse': ['baseline_attention', 'previous_F', 'B3-D02'], 'new': 'all remaining methods including F0'})
    plan = json.loads((ORIGINAL / 'runplan.json').read_text())
    tasks = []
    for fold in (1, 2, 3):
        (FOLDER / 'scores' / f'fold{fold}').mkdir(parents=True, exist_ok=True)
        request_path = FOLDER / f'score_request_fold{fold}.json'
        request = json.loads(request_path.read_text())
        inference = json.loads((FOLDER / f'inference_fold{fold}.json').read_text())
        assert all(sha(Path(p)) == inference['predictions_sha256'][name] for name, p in request['predictions'].items())
        offset = 0
        for item in plan['groups'][plan['folds'][fold - 1]['evaluation']]:
            tasks.append((fold, Path(item['file']).stem, offset, sha(request_path)))
            offset += item['scenes']
        assert offset == inference['n'] == len(request['ids'])
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(score_file, tasks))
    for fold in (1, 2, 3):
        paths = [Path(p) for p in results if Path(p).parent.name == f'fold{fold}']
        total, seen = None, set()
        for path in paths:
            with np.load(path) as values:
                names = values['methods'].tolist()
                ids = values['sample_ids'].tolist()
                assert not seen.intersection(ids) and len(set(ids)) == len(ids)
                seen.update(ids)
                sums = np.column_stack((values['ADE'].sum(1), values['FDE'].sum(1), values['collision'][:, :, 0].sum(1), values['collision'][:, :, 1].sum(1)))
                total = sums if total is None else total + sums
        assert total is not None
        total /= len(seen)
        summary = {name: {'n': len(seen), 'ADE': float(v[0]), 'FDE': float(v[1]), 'ColI_all': float(v[2] * 100), 'ColI_obs': float(v[3] * 100)} for name, v in zip(names, total)}
        folder = FOLDER / 'scores' / f'fold{fold}'
        save(folder / 'summary.json', summary)
        save(folder / 'receipt.json', {'n': len(seen), 'missing': 0, 'duplicates': 0, 'outputs': {p.name: sha(p) for p in [*paths, folder / 'summary.json']}})


if __name__ == '__main__':
    main()
