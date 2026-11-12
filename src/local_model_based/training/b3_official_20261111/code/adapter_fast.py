"""현재 fold의 관측과 학습 자료만 사용하는 FAST/TRM 실행기."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from adapter_io import ROOT, anchor, load_role, sha

KINDS = ('continuous', 'raw_bpe', 'dct_bpe', 'two_state_deep', 'two_state_last', 'single_state')
SEED = 20261110


def model_module(family):
    path = ROOT / 'adapters' / family / 'model.py'
    spec = importlib.util.spec_from_file_location('isolated_' + family, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def guard(family, stage, fold):
    folder = ROOT / 'adapters' / family
    tmp = folder / '.tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)
    os.environ['TMPDIR'] = str(tmp)
    plan = json.loads((ROOT / 'runplan.json').read_text())
    allowed_targets = set()
    if stage in ('train', 'probe'):
        for key in ('train', 'development'):
            for row in plan['groups'][plan['folds'][fold - 1][key]]:
                allowed_targets.add((ROOT / 'data/targets' / (Path(row['file']).stem + '_labels.npz')).resolve())

    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto', 'subprocess.Popen', 'os.system'):
            raise PermissionError('network_or_subprocess_blocked')
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            if not path.is_relative_to(folder) and path != Path('/dev/null'):
                raise PermissionError('outside_adapter_write_blocked: ' + str(path))
        if path.is_relative_to(ROOT / 'data/targets') and path not in allowed_targets:
            raise PermissionError('future_supervision_blocked: ' + str(path))
        if path.suffix == '.ndjson':
            raise PermissionError('raw_future_source_blocked')
        if path.is_relative_to(Path('/mnt/RG2')) and not path.is_relative_to(ROOT) and not path.is_relative_to(Path(sys.prefix)):
            raise PermissionError('historical_cache_or_source_blocked: ' + str(path))
    sys.addaudithook(audit)


def normalized(batch, values):
    return np.matmul(values - batch.origin[:, None], batch.rotation) / batch.scale[:, None, None]


def features(batch, base, anchored):
    parts = [batch.ego.reshape(len(batch.ego), -1), (batch.neighbors * batch.mask[:, :, None]).reshape(len(batch.ego), -1), batch.mask]
    if anchored:
        parts.append(normalized(batch, base).reshape(len(batch.ego), -1))
    return np.concatenate(parts, axis=1)


def world(batch, values, base, anchored):
    origin = base if anchored else batch.origin[:, None]
    return origin + values @ batch.rotation.transpose(0, 2, 1) * batch.scale[:, None, None]


def role_signature(fold, role, metadata, targets):
    plan = json.loads((ROOT / 'runplan.json').read_text())
    group = plan['folds'][fold - 1][{'train': 'train', 'dev': 'development', 'eval': 'evaluation'}[role]]
    receipts = {Path(row['file']).stem: sha(ROOT / 'data/receipts' / (Path(row['file']).stem + '.json')) for row in plan['groups'][group]}
    ap = ROOT / 'attention' / f'fold{fold}' / f'direct_unshared4_{role}.npy'
    return {'role': role, 'n': len(metadata), 'order_sha256': digest(metadata), 'data_receipts': receipts, 'anchor_sha256': sha(ap), 'targets_loaded': targets}


def signature(family, fold, roles):
    files = [Path(__file__), ROOT / 'code/adapter_io.py', ROOT / 'code/features.py', ROOT / 'adapters' / family / 'model.py']
    if family == 'masked':
        files.append(ROOT / 'code/adapter_masked.py')
    return {'runplan': sha(ROOT / 'runplan.json'), 'source': {str(p.relative_to(ROOT)): sha(p) for p in files}, 'fold': fold, 'family': family, 'roles': roles, 'seed': SEED, 'optimizer': 'AdamW', 'lr': .001, 'weight_decay': .01, 'batch_size': 256, 'max_epochs': 25, 'patience': 6, 'minimum_epochs': 5, 'checkpoint_selection': 'minimum_dev_physical_ADE', 'final_gate': 'external_controller_ADE_FDE_ColII', 'tf32': False}


def predict_offsets(module, net, x, codec=None):
    if codec is not None or hasattr(net, 'continuous'):
        return module.predict(net, x, codec)
    net.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), 256):
            outputs.append(net.predict(torch.as_tensor(x[start:start + 256], dtype=torch.float32, device='cuda')).cpu().numpy())
    return np.concatenate(outputs)


def prepare_codecs(module, folder, datasets, fingerprint):
    receipt_path = folder / 'codec_receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['fingerprint'] == fingerprint, 'codec_signature_changed'
        assert all(sha(folder / p) == h for p, h in receipt['artifacts'].items())
        return receipt['scale']
    rows = []
    for scale in (10., 20., 40.):
        if scale == 40. and not any(r['scale'] == 20 and r['frequency'] and r['role'] == 'dev' and r['ADE'] > .05 for r in rows):
            break
        for name, (train_y, dev_y, train_scale, dev_scale) in datasets.items():
            for frequency in (False, True):
                codec = module.fit_codec(train_y, scale, frequency)
                codec.persist(folder / f'{name}_{"dct" if frequency else "raw"}_q{int(scale)}')
                for role, values, physical in (('train', train_y, train_scale), ('dev', dev_y, dev_scale)):
                    tokens = codec.encode(values)
                    error = np.linalg.norm(codec.decode(tokens) - values, axis=-1) * physical[:, None]
                    rows.append({'anchor': name, 'scale': int(scale), 'frequency': frequency, 'role': role, 'ADE': float(error.mean()), 'FDE': float(error[:, -1].mean()), 'mean_tokens': float(np.mean(list(map(len, tokens)))), 'clipped_fraction': codec.quantized(values)[1]})
    chosen = max(r['scale'] for r in rows)
    artifacts = {p.name: sha(p) for p in folder.glob('*_q*.*')}
    save(receipt_path, {'fingerprint': fingerprint, 'scale': chosen, 'rule': 'q20 우선, dev DCT 복원 ADE>.05이면 사전 규정 q40만 추가', 'rows': rows, 'artifacts': artifacts, 'fit_source': 'train_only', 'prediction_metric': False})
    return chosen


def fit_one(module, family, folder, name, kind, data, base_signature, quantization, probe):
    tr, dv, ty, dy, tb, db = data
    anchored = name.startswith('local_')
    x = features(tr, tb, anchored)
    dx = features(dv, db, anchored)
    y = normalized(tr, ty) - (normalized(tr, tb) if anchored else 0)
    dev_y = normalized(dv, dy) - (normalized(dv, db) if anchored else 0)
    center, spread = x.mean(0), np.maximum(x.std(0), .05)
    x, dx = np.clip((x - center) / spread, -10, 10), np.clip((dx - center) / spread, -10, 10)
    spec = dict(base_signature, name=name, probe=probe, quantization=quantization)
    checkpoint = folder / (name + '.pt')
    receipt_path = folder / (name + '.receipt.json')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['signature'] == spec, 'fit_signature_changed'
        assert sha(checkpoint) == receipt['checkpoint_sha256'], 'checkpoint_changed'
        assert sha(folder / 'dev' / (name + '.npy')) == receipt['prediction_sha256'], 'dev_prediction_changed'
        print(json.dumps({'resumed': name, 'fold': spec['fold']}), flush=True)
        return
    started = time.monotonic()
    torch.manual_seed(SEED)
    codec_name = f'{name.split("_")[0]}_{kind.split("_")[0]}_q{quantization}' if kind.endswith('bpe') else None
    codec = module.Codec.restore(folder / codec_name) if codec_name else None
    net = module.Predictor(x.shape[1], kind, codec.tokenizer.get_vocab_size() if codec else 384).cuda() if family == 'fast' else module.Predictor(x.shape[1], kind).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
    xt, yt, st = [torch.as_tensor(a, dtype=torch.float32, device='cuda') for a in (x, y, tr.scale)]
    if codec:
        tokens = np.zeros((len(y), 24), dtype=np.int64)
        for i, seq in enumerate(codec.encode(y)):
            tokens[i, :len(seq)] = seq
        targets = torch.as_tensor(tokens, device='cuda')
        prefix = torch.cat((torch.ones((len(y), 1), dtype=torch.long, device='cuda'), targets[:, :-1]), dim=1)
    best, stale, steps, curve = float('inf'), 0, 0, []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, 3 if probe else 26):
        net.train()
        losses = []
        for ids in torch.randperm(len(x), device='cuda').split(256):
            optimizer.zero_grad(set_to_none=True)
            if family == 'masked':
                loss = net.objective(xt[ids], yt[ids], st[ids])
            elif codec:
                logits = net.teacher(xt[ids], prefix[ids])
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets[ids].reshape(-1), ignore_index=0)
            else:
                output = net.continuous(xt[ids])
                used = output if kind == 'two_state_deep' else output[-1:]
                loss = sum((torch.linalg.vector_norm(value.reshape(-1, 12, 2) - yt[ids], dim=-1) * st[ids, None]).mean() for value in used) / len(used)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
        prediction = predict_offsets(module, net, dx, codec)
        error = np.linalg.norm(prediction - dev_y, axis=-1) * dv.scale[:, None]
        row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'ADE': float(error.mean()), 'FDE': float(error[:, -1].mean())}
        curve.append(row)
        if row['ADE'] < best:
            best, stale = row['ADE'], 0
            torch.save({'state': net.state_dict(), 'kind': kind, 'center': center, 'spread': spread, 'inputs': x.shape[1], 'codec': codec_name, 'vocabulary': codec.tokenizer.get_vocab_size() if codec else 384, 'signature': spec}, checkpoint)
        else:
            stale += 1
        if epoch >= 5 and stale >= 6:
            break
    ck = torch.load(checkpoint, map_location='cuda', weights_only=False)
    net.load_state_dict(ck['state'])
    values = world(dv, predict_offsets(module, net, dx, codec), db, anchored)
    assert values.shape == (len(dx), 12, 2) and np.isfinite(values).all()
    (folder / 'dev').mkdir(exist_ok=True)
    output = folder / 'dev' / (name + '.npy')
    np.save(output, values)
    receipt = {'signature': spec, 'checkpoint_sha256': sha(checkpoint), 'prediction_sha256': sha(output), 'seconds': time.monotonic() - started, 'curve': curve, 'optimizer_steps': steps, 'parameters': sum(p.numel() for p in net.parameters()), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'), 'api_calls': 0, 'train_n': len(x), 'dev_n': len(dx)}
    save(receipt_path, receipt)
    print(json.dumps({'method': name, 'seconds': receipt['seconds'], 'epochs': len(curve), 'train_n': len(x), 'dev_n': len(dx)}), flush=True)


def train(family, fold, kinds, probe=False):
    module = model_module(family)
    tr, tm, ty = load_role(fold, 'train', targets=True)
    dv, dm, dy = load_role(fold, 'dev', targets=True)
    tb, db = anchor(fold, 'train'), anchor(fold, 'dev')
    assert tb.shape == ty.shape and db.shape == dy.shape
    roles = [role_signature(fold, 'train', tm, True), role_signature(fold, 'dev', dm, True)]
    if probe:
        tr, ty, tb = tr.take(np.arange(min(256, len(ty)))), ty[:256], tb[:256]
        dv, dy, db = dv.take(np.arange(min(128, len(dy)))), dy[:128], db[:128]
    folder = ROOT / 'adapters' / family / (f'probe_fold{fold}' if probe else f'fold{fold}')
    folder.mkdir(parents=True, exist_ok=True)
    base_signature = signature(family, fold, roles)
    initial = folder / 'input_signature.json'
    if initial.exists():
        assert json.loads(initial.read_text()) == base_signature, 'input_or_code_changed'
    else:
        save(initial, base_signature)
    quantization = None
    if family == 'fast':
        datasets = {}
        for name, anchored in (('direct', False), ('local', True)):
            datasets[name] = (normalized(tr, ty) - (normalized(tr, tb) if anchored else 0), normalized(dv, dy) - (normalized(dv, db) if anchored else 0), tr.scale, dv.scale)
        quantization = prepare_codecs(module, folder, datasets, digest(base_signature))
    for name in ('direct', 'local'):
        for kind in kinds:
            fit_one(module, family, folder, name + '_' + kind, kind, (tr, dv, ty, dy, tb, db), base_signature, quantization, probe)
    assert signature(family, fold, roles) == base_signature, 'source_changed_during_training'
    save(folder / 'training_receipt.json', {'fits': len(kinds) * 2, 'probe': probe, 'signature': base_signature, 'api_calls': 0, 'selection_pending_controller': True})


def infer(family, fold, role, kinds):
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    module = model_module(family)
    batch, metadata, labels = load_role(fold, role, targets=False)
    assert labels is None
    base = anchor(fold, role)
    assert base.shape == (len(batch.ego), 12, 2)
    folder = ROOT / 'adapters' / family / f'fold{fold}'
    frozen = json.loads((folder / 'input_signature.json').read_text())
    assert frozen['runplan'] == sha(ROOT / 'runplan.json')
    for path, h in frozen['source'].items():
        assert sha(ROOT / path) == h, 'source_changed'
    role_receipt = role_signature(fold, role, metadata, False)
    outputs = {}
    for name in ('direct', 'local'):
        x = features(batch, base, name == 'local')
        for kind in kinds:
            method = name + '_' + kind
            checkpoint = folder / (method + '.pt')
            receipt = json.loads(checkpoint.with_suffix('.receipt.json').read_text())
            assert sha(checkpoint) == receipt['checkpoint_sha256']
            ck = torch.load(checkpoint, map_location='cuda', weights_only=False)
            assert ck['signature'] == receipt['signature']
            codec = module.Codec.restore(folder / ck['codec']) if ck['codec'] else None
            if codec:
                cr = json.loads((folder / 'codec_receipt.json').read_text())
                for ext in ('.json', '.npz'):
                    assert sha(folder / (ck['codec'] + ext)) == cr['artifacts'][ck['codec'] + ext]
            net = module.Predictor(ck['inputs'], kind, ck['vocabulary']).cuda() if family == 'fast' else module.Predictor(ck['inputs'], kind).cuda()
            net.load_state_dict(ck['state'])
            nx = np.clip((x - ck['center']) / ck['spread'], -10, 10)
            values = world(batch, predict_offsets(module, net, nx, codec), base, name == 'local')
            assert values.shape == (len(metadata), 12, 2) and np.isfinite(values).all()
            output = folder / role / (method + '.npy')
            output.parent.mkdir(exist_ok=True)
            if output.exists():
                assert np.array_equal(np.load(output), values), 'replay_changed'
            else:
                np.save(output, values)
            outputs[method] = sha(output)
    save(folder / (role + '_inference_receipt.json'), {'role': role_receipt, 'outputs': outputs, 'future_targets_loaded': False, 'api_calls': 0, 'seconds': time.monotonic() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')})
    print(json.dumps({'family': family, 'fold': fold, 'role': role, 'n': len(metadata), 'methods': len(outputs)}), flush=True)


def main(family='fast', kinds=KINDS):
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['probe', 'train', 'predict-dev', 'predict-eval'])
    parser.add_argument('fold', type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    guard(family, args.stage, args.fold)
    if args.stage in ('train', 'probe'):
        train(family, args.fold, kinds, probe=args.stage == 'probe')
    else:
        infer(family, args.fold, args.stage.removeprefix('predict-'), kinds)


if __name__ == '__main__':
    main()
