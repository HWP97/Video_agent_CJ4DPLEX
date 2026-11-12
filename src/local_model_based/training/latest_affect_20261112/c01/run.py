# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# How to run: 기존 b1 Python 환경으로 python -B run.py
"""b2의 내부 선택, 그룹 OOF 재학습, 관측 전용 전수 예측을 실행한다."""
from pathlib import Path
from dataclasses import asdict
import json
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from followup_common import ROOT, SEED, sha, save, freeze, guard, metric
from models import CONFIGS, Expert, Combiner
from engine import Data, Fit, oof, combine

FOLDER = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'official_local_train_20261111/b2/attempt_fp32'


def load_data() -> Data:
    rows = json.loads((SOURCE / 'manifest.json').read_text())
    with np.load(SOURCE / 'inputs.npz') as original:
        raw, numeric = original['x'], original['numeric']
        hold = np.repeat(original['recognition'][:, -1:, :], 4, 1).astype(np.float32)
        smooth = np.repeat((original['recognition'][:, -3:, :] * np.array([.2, .3, .5])[None, :, None]).sum(1)[:, None], 4, 1).astype(np.float32)
    groups, splits = np.array([r['group'] for r in rows]), np.array([r['split'] for r in rows])
    labels = {r['sample_id']: r['future'] for r in json.loads((SOURCE / 'labels_train.json').read_text())}
    y = np.full((len(rows), 4, 2), np.nan, np.float32)
    train = np.flatnonzero(splits == 'train')
    assert len(train) == 1169
    y[train] = np.array([labels[rows[i]['sample_id']] for i in train], np.float32)
    times = np.array([r['frame_pts_s'] for r in rows], np.float32)
    return Data(raw, numeric, (times - times[:, -1:]) / 8, hold, smooth, y, groups, splits)


def stage(data: Data, specification: tuple):
    name, train, predict, seed, configs = specification
    folder = FOLDER / name / str(seed)
    experts = {mtp: oof(data, Fit(folder / ('M' if mtp else 'A'), train, predict, seed, mtp)) for mtp in sorted({c.mtp for c in configs})}
    outputs, scores = {}, {}
    for config in configs:
        (p, obs), (ep, ef) = experts[config.mtp]
        hold = data.smooth if config.smooth else data.hold
        arrays = ((p, hold[train], obs, data.y[train]), (ep, hold[predict], ef))
        values, gates = combine(config, folder / config.name, arrays, seed)
        output = folder / config.name / 'prediction.npz'
        if output.exists():
            with np.load(output) as old:
                assert np.array_equal(old['prediction'], values)
        else:
            np.savez(output, prediction=values, gates=gates, ids=predict)
        outputs[config.name] = str(output.relative_to(FOLDER))
        if name == 'inner':
            scores[config.name] = metric(values, data.y[predict])
    save(folder / 'receipt.json', {'outputs': {p: sha(FOLDER / p) for p in outputs.values()}, 'scores': scores,
         'fit_ids': train.tolist(), 'inference_ids': predict.tolist(), 'seed': seed, 'future_GT_inference': False})
    return outputs, scores


def main() -> None:
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    guard(FOLDER, True)
    data = load_data()
    with np.load(FOLDER.parent / 'data/training.npz') as split:
        train_ids = np.flatnonzero(data.splits == 'train')
        inner, validation = train_ids[split['inner_train']], train_ids[split['inner_validation']]
    assert len(inner) == 913 and len(validation) == 256
    assert not set(data.groups[inner]) & set(data.groups[validation])
    # 실제 입력에서 추론 API에 target이 없고 C identity·gate=0 복귀가 성립한다.
    model = Expert().eval()
    with torch.inference_mode():
        p = model.predict(torch.zeros(3, 15, 33), torch.zeros(3, 6))
        combiner = Combiner(CONFIGS[3])
        hold = torch.from_numpy(data.hold[:3])
        identity = combiner((p, hold, torch.zeros(3, 6)))[0]
        assert torch.allclose(identity, p.clamp(-1, 1), atol=1e-7, rtol=0)
    freeze(FOLDER / 'registration.json', {'configs': [asdict(c) for c in CONFIGS], 'seed': SEED,
        'sources': {str(p): sha(p) for p in [*FOLDER.glob('*.py'), ROOT / 'followup_common.py']},
        'inputs': {str(SOURCE / n): sha(SOURCE / n) for n in ('inputs.npz', 'manifest.json', 'labels_train.json')},
        'split': sha(FOLDER.parent / 'data/training.npz'), 'selection_rule': 'hold 대비 MAE<= 및 CCC>= 중 MAE 최소, 없으면 hold',
        'expert_epochs': 20, 'combiner_epochs': 100, 'optional_followups': 0, 'small_batch_shape': list(p.shape)})
    _, scores = stage(data, ('inner', inner, validation, SEED, CONFIGS))
    baseline = metric(data.hold[validation], data.y[validation])
    passing = [n for n, s in scores.items() if s['MAE'] <= baseline['MAE'] and s['CCC'] >= baseline['CCC']]
    selected = min(passing, key=lambda n: scores[n]['MAE']) if passing else 'baseline_hold'
    seeds = [SEED]
    if scores['B2-C08']['MAE'] <= scores['B2-C01']['MAE'] and scores['B2-C08']['CCC'] >= scores['B2-C01']['CCC'] and scores['B2-C08'] != scores['B2-C01']:
        seeds += [20261112, 20261113]
    freeze(FOLDER / 'selection_lock.json', {'selected': selected, 'baseline': baseline, 'inner_scores': scores,
         'additional_seeds': seeds[1:], 'optional_followups': [], 'outer_labels_read': False,
         'registration': sha(FOLDER / 'registration.json'), 'inner_receipt': sha(FOLDER / 'inner' / str(SEED) / 'receipt.json')})
    all_outputs = {}
    for seed in seeds:
        configs = CONFIGS if seed == SEED else (CONFIGS[0], CONFIGS[7])
        if seed != SEED:
            stage(data, ('inner', inner, validation, seed, configs))
        outputs, _ = stage(data, ('full', train_ids, np.arange(len(data.raw)), seed, configs))
        all_outputs[str(seed)] = outputs
    save(FOLDER / 'inference_receipt.json', {'seeds': all_outputs, 'registration': sha(FOLDER / 'registration.json'),
         'selection_lock': sha(FOLDER / 'selection_lock.json'), 'GT_read': False, 'API_calls': 0})


if __name__ == '__main__':
    main()
