# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scipy", "tokenizers"]
# ///
# How to run: CUDA_VISIBLE_DEVICES=0 python -B infer.py
"""모든 fold 선택 동결 뒤 관측만으로 예측하고 재생을 검증한다."""
from pathlib import Path
import json
import sys
import importlib.util
import numpy as np
import torch
helper_path = Path(__file__).resolve().parent / 'common.py'
helper_spec = importlib.util.spec_from_file_location('targeted_experiment_common', helper_path)
assert helper_spec is not None and helper_spec.loader is not None
helper_module = importlib.util.module_from_spec(helper_spec)
sys.modules[helper_spec.name] = helper_module
helper_spec.loader.exec_module(helper_module)
from targeted_experiment_common import (FOLDER, ROOT, OLD, ORIGINAL, SEEDS, Model, CONFIGS,
                                        predict, setup, sha, save, freeze, ids_sha)
sys.path.insert(0, str(ORIGINAL / 'code'))
from adapter_io import load_role, install_guard


def main() -> None:
    setup()
    freeze(FOLDER / 'inference_source_lock.json', {str(p): sha(p) for p in [Path(__file__), FOLDER / 'common.py', OLD / 'models.py', OLD / 'engine.py', ORIGINAL / 'adapters/fast/model.py', ORIGINAL / 'code/adapter_io.py']})
    frozen = json.loads((FOLDER / 'all_folds_frozen.json').read_text())
    assert all(sha(FOLDER / f'selection_fold{f}.json') == frozen[str(f)] for f in (1, 2, 3))
    for fold in (1, 2, 3):
        install_guard('infer', fold)
        lock = json.loads((FOLDER / f'selection_fold{fold}.json').read_text())
        observations, rows, labels = load_role(fold, 'eval', targets=False)
        assert labels is None
        count = len(rows)
        kalman = ((observations.kalman - observations.origin[:, None]) @ observations.rotation / observations.scale[:, None, None]).astype(np.float32)
        data = {k: torch.as_tensor(v, device='cuda', dtype=torch.float32) for k, v in {'ego': observations.ego, 'neighbors': observations.neighbors, 'mask': observations.mask, 'kalman': kalman}.items()}
        methods = {}
        for seed in SEEDS:
            for policy in ('robust', 'single'):
                epoch = lock['seeds'][str(seed)][policy + '_epoch']
                methods[f'F_{policy}_seed{seed}'] = FOLDER / 'full' / f'fold{fold}' / str(seed) / 'F' / f'epoch{epoch}.pt'
        for name in ('B3-D03', 'B3-D04'):
            methods[name] = FOLDER / 'full' / f'fold{fold}' / name / 'best.pt'
        outputs, replay = {}, []
        deduplicated = {}
        folder = FOLDER / 'predictions' / f'fold{fold}'
        folder.mkdir(parents=True, exist_ok=True)
        for name, checkpoint in methods.items():
            if checkpoint in deduplicated:
                outputs[name] = deduplicated[checkpoint]
                continue
            saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
            data['center'], data['spread'] = saved['center'], saved['spread']
            model = Model(CONFIGS[0]).cuda().eval()
            model.load_state_dict(saved['state'])
            values, _ = predict(model, data, np.arange(count))
            normalized = values[:, -1].astype(np.float64)
            world = observations.origin[:, None] + normalized @ observations.rotation.transpose(0, 2, 1) * observations.scale[:, None, None]
            path = folder / f'{name}.npy'
            if path.exists():
                assert np.array_equal(np.load(path), world)
            else:
                np.save(path, world)
            again, _ = predict(model, data, np.arange(min(512, count)))
            difference = float(np.max(np.abs(values[:len(again)] - again)))
            assert difference == 0
            outputs[name] = str(path)
            deduplicated[checkpoint] = str(path)
            replay.append({'method': name, 'n': count, 'replay_n': len(again), 'maxdiff': difference,
                           'prediction_sha256': sha(path), 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint), 'epoch': saved['epoch'], 'GT_read': False})
            print(json.dumps({'fold': fold, 'method': name, 'n': count}), flush=True)
        diagnostic = ROOT / 'diagnosis_followup/b3' / f'fold{fold}_F_before_finetune.npy'
        record = json.loads(diagnostic.with_suffix('.json').read_text())
        assert sha(diagnostic) == record['prediction_sha256'] and len(np.load(diagnostic, mmap_mode='r')) == count
        assert sha(OLD / 'full' / f'fold{fold}' / '20261111/F/best.pt') == record['checkpoint_sha256']
        outputs['B3-D01'] = str(diagnostic)
        epoch0_preserved = []
        for name in ('B3-D03', 'B3-D04'):
            if lock['additional_epochs'][name] == 0:
                before = torch.load(OLD / 'full' / f'fold{fold}' / '20261111/F/best.pt', map_location='cpu', weights_only=False)
                after = torch.load(methods[name], map_location='cpu', weights_only=False)
                assert all(torch.equal(before['state'][k], after['state'][k]) for k in before['state'])
                assert torch.equal(before['center'], after['center']) and torch.equal(before['spread'], after['spread'])
                difference = float(np.max(np.abs(np.load(outputs[name], mmap_mode='r') - np.load(diagnostic, mmap_mode='r'))))
                epoch0_preserved.append({'method': name, 'parent_state_and_transform_bitwise_equal': True,
                       'historical_F0_maxdiff': difference, 'historical_F0_bitwise_equal': difference == 0})
        old_request = json.loads((OLD / f'score_request_fold{fold}.json').read_text())
        assert old_request['ids'] == list(range(count))
        outputs['B3-D02'] = old_request['predictions']['B3-C01']
        for name in ('baseline_attention', 'previous_F'):
            outputs[name] = old_request['predictions'][name]
        manifest = json.loads((ORIGINAL / 'prediction_manifests' / f'fold{fold}_eval.json').read_text())
        for name, key in (('baseline_attention', 'loop/direct_unshared4_d4'), ('previous_F', 'fast_trm/direct_two_state_deep')):
            assert sha(Path(outputs[name])) == manifest['verified_predictions'][key]['sha256']
        prior = json.loads((OLD / f'inference_fold{fold}.json').read_text())
        old_record = next(r for r in prior['replay'] if r['method'] == 'B3-C01')
        assert sha(Path(outputs['B3-D02'])) == old_record['prediction_sha256']
        # 원본 index 순서와 adapter 반환 순서를 직접 비교한다.
        plan = json.loads((ORIGINAL / 'runplan.json').read_text())
        index_rows = []
        for item in plan['groups'][plan['folds'][fold - 1]['evaluation']]:
            index_rows.extend(json.loads((ORIGINAL / 'data/indices' / (Path(item['file']).stem + '_index.json')).read_text()))
        assert [r['sample_id'] for r in rows] == [r['sample_id'] for r in index_rows]
        save(FOLDER / f'inference_fold{fold}.json', {'outputs': outputs, 'replay': replay, 'n': count, 'GT_read': False,
             'epoch0_full_preservation_vs_historical_F0': epoch0_preserved,
             'source': sha(Path(__file__)), 'order_ids_sha256': ids_sha(np.array([r['sample_id'] for r in rows])),
             'predictions_sha256': {n: sha(Path(p)) for n, p in outputs.items()}})
        freeze(FOLDER / f'score_request_fold{fold}.json', {'destination': str(FOLDER / 'scores' / f'fold{fold}'),
               'fold': fold, 'role': 'eval', 'ids': list(range(count)), 'predictions': outputs})


if __name__ == '__main__':
    main()
