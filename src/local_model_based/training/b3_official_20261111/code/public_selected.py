"""개발 선택 모델을 재학습 없이 public 관측에 적용한다."""
from __future__ import annotations
import json
import sys
import numpy as np
import torch
from adapter_io import ROOT, anchor, load_role, sha
from features import load
from run_guarded import install


def main() -> None:
    fold = int(sys.argv[1])
    lock = json.loads((ROOT / 'all_fold_selections_frozen.json').read_text())
    selection_path = ROOT / 'selections' / f'fold{fold}.json'
    assert lock['folds'][selection_path.name] == sha(selection_path)
    selection = json.loads(selection_path.read_text())
    method = selection['ours']['selected']
    install('infer', fold)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    public = ROOT / 'public'
    preparation = json.loads((public / 'preparation_receipt.json').read_text())
    for name, digest in preparation['outputs'].items():
        assert sha(public / name) == digest
    batch = load(public / 'inputs.npz')
    dev, metadata, labels = load_role(fold, 'dev', targets=False)
    assert labels is None
    dev = dev.take(list(range(256)))
    family, name = method.split('/', 1)
    match family:
        case 'fast_trm':
            from adapter_fast import features, model_module, predict_offsets, world
            module = model_module('fast')
            folder = ROOT / 'adapters/fast' / f'fold{fold}'
            frozen = json.loads((folder / 'input_signature.json').read_text())
            for path, digest in frozen['source'].items():
                assert sha(ROOT / path) == digest
            path = folder / (name + '.pt')
            receipt = json.loads(path.with_suffix('.receipt.json').read_text())
            assert sha(path) == receipt['checkpoint_sha256']
            ck = torch.load(path, map_location='cuda', weights_only=False)
            assert ck['signature'] == receipt['signature']
            codec = module.Codec.restore(folder / ck['codec']) if ck['codec'] else None
            if codec:
                cr = json.loads((folder / 'codec_receipt.json').read_text())
                for ext in ('.json', '.npz'):
                    assert sha(folder / (ck['codec'] + ext)) == cr['artifacts'][ck['codec'] + ext]
            net = module.Predictor(ck['inputs'], ck['kind'], ck['vocabulary']).cuda()
            net.load_state_dict(ck['state'])
            anchored = name.startswith('local_')

            def infer(data, base):
                x = features(data, base, anchored)
                nx = np.clip((x - ck['center']) / ck['spread'], -10, 10)
                return world(data, predict_offsets(module, net, nx, codec), base, anchored)

            replay = infer(dev, anchor(fold, 'dev')[:256])
            expected = np.load(folder / 'dev' / (name + '.npy'))[:256]
            values = infer(batch, np.load(public / 'predictions' / f'baseline_fold{fold}.npy'))
        case 'residual':
            from adapter_residual import Trial, create_model, observed_features, predict_residual, sources
            folder = ROOT / 'adapters/residual' / f'fold{fold}'
            path = folder / (name + '.pt')
            receipt = json.loads(path.with_suffix('.receipt.json').read_text())
            assert sha(path) == receipt['checkpoint_sha256']
            assert receipt['signature']['sources'] == sources()
            ck = torch.load(path, map_location='cuda', weights_only=False)
            assert ck['signature'] == receipt['signature']
            trial = Trial(**ck['trial'])
            net = create_model(trial, ck['center'], ck['spread'], 'cuda')
            net.load_state_dict(ck['state_dict'])
            base = anchor(fold, 'dev')[:256] if trial.baseline == 'attention' else getattr(dev, trial.baseline)
            inputs = observed_features(dev, base, np.array([1 / row['fps'] for row in metadata[:256]]))
            replay = predict_residual(net, inputs, base, trial, 'cuda')
            expected = np.load(folder / 'dev' / (name + '.npy'))[:256]
            base = np.load(public / 'predictions' / f'baseline_fold{fold}.npy') if trial.baseline == 'attention' else getattr(batch, trial.baseline)
            index = json.loads((public / 'index.json').read_text())
            inputs = observed_features(batch, base, np.array([1 / row['fps'] for row in index]))
            values = predict_residual(net, inputs, base, trial, 'cuda')
        case 'memory_ttt':
            from adapter_memory import predict_memory, source_signature
            folder = ROOT / 'adapters/memory' / f'fold{fold}'
            path = folder / 'memory.npz'
            receipt = json.loads((folder / 'training.json').read_text())
            assert sha(path) == receipt['checkpoint_sha256']
            assert receipt['signature'] == source_signature(fold, 'train')
            with np.load(path) as saved:
                memory = {key: saved[key] for key in saved.files}
            replay = predict_memory(dev, anchor(fold, 'dev')[:256],
                np.array([1 / row['fps'] for row in metadata[:256]]), memory)[name]
            expected = np.load(folder / 'dev' / (name + '.npy'))[:256]
            index = json.loads((public / 'index.json').read_text())
            values = predict_memory(batch, np.load(public / 'predictions' / f'baseline_fold{fold}.npy'),
                np.array([1 / row['fps'] for row in index]), memory)[name]
        case 'loop':
            from model import Config, Recurrent, predict
            config, depth = name.rsplit('_d', 1)
            path = ROOT / 'attention' / f'fold{fold}' / (config + '.pt')
            receipt = json.loads(path.with_suffix('.json').read_text())
            assert sha(path) == receipt['checkpoint_sha256']
            ck = torch.load(path, map_location='cuda', weights_only=False)
            net = Recurrent(Config(**ck['config'])).cuda()
            net.load_state_dict(ck['state_dict'])
            replay = predict(net, dev, int(depth))
            expected = np.load(path.parent / f'{config}_dev_d{depth}.npy')[:256]
            values = predict(net, batch, int(depth))
        case unknown:
            raise RuntimeError('선택 family의 public 추론 연결 필요: ' + unknown)
    difference = float(np.abs(replay - expected).max())
    assert difference == 0, (method, difference)
    assert values.shape == (3146, 12, 2) and np.isfinite(values).all()
    output = public / 'predictions' / f'ours_fold{fold}.npy'
    assert not output.exists()
    np.save(output, values)
    result = {'fold': fold, 'method': method, 'selection_sha256': sha(selection_path),
        'checkpoint_sha256': sha(path), 'prediction_sha256': sha(output),
        'input_sha256': sha(public / 'inputs.npz'), 'source_sha256': sha(__file__),
        'dev_replay_256_maxdiff': difference, 'samples': 3146, 'new_fit': False,
        'future_targets_loaded': False, 'API_calls': 0}
    (public / f'ours_fold{fold}_receipt.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
