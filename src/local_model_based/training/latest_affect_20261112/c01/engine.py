# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# How to run: run.py에서 import한다.
"""그룹별 transform과 전문가 fit, OOF 재사용을 관리한다."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from models import Expert, Combiner, Config, ccc_loss
from followup_common import sha, save, tensor_checkpoint, metric


@dataclass(frozen=True, slots=True)
class Data:
    raw: np.ndarray
    numeric: np.ndarray
    times: np.ndarray
    hold: np.ndarray
    smooth: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    splits: np.ndarray


@dataclass(frozen=True, slots=True)
class Fit:
    folder: Path
    train: np.ndarray
    predict: np.ndarray
    seed: int
    mtp: bool


def expert_fit(data: Data, fit: Fit):
    """예측 대상 그룹을 제외한 학습 부분에서만 PCA·통계를 fit한다."""
    folder = fit.folder
    folder.mkdir(parents=True, exist_ok=True)
    signature = {'train': fit.train.tolist(), 'predict': fit.predict.tolist(), 'seed': fit.seed,
                 'mtp': fit.mtp, 'epochs': 20, 'sources': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
                 'train_groups': sorted(set(data.groups[fit.train].tolist()))}
    receipt_path = folder / 'receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['signature'] == signature
        assert all(sha(folder / p) == h for p, h in receipt['outputs'].items())
        with np.load(folder / 'predictions.npz') as saved:
            return saved['expert'], saved['observed']
    torch.manual_seed(fit.seed)
    transform_path = folder / 'transform.npz'
    if not transform_path.exists():
        flat = torch.as_tensor(data.raw[fit.train].reshape(-1, data.raw.shape[-1]), device='cuda')
        mean, scale = flat.mean(0), flat.std(0, unbiased=False).clamp_min(1e-4)
        _, _, basis = torch.pca_lowrank((flat - mean) / scale, q=40, center=False, niter=3)
        nm, ns = data.numeric[fit.train].mean(0), np.maximum(data.numeric[fit.train].std(0), 1e-4)
        np.savez(transform_path, mean=mean.cpu().numpy(), scale=scale.cpu().numpy(), basis=basis[:, :32].cpu().numpy(), nm=nm, ns=ns)
        del flat
    with np.load(transform_path) as t:
        transformed = [((data.raw[i:i + 128] - t['mean']) / t['scale']) @ t['basis'] / np.sqrt(32) for i in range(0, len(data.raw), 128)]
        x = np.concatenate((np.concatenate(transformed), data.times[:, :, None]), -1).astype(np.float32)
        numeric = ((data.numeric - t['nm']) / t['ns']).astype(np.float32)
    x, numeric = torch.from_numpy(x), torch.from_numpy(numeric)
    target = torch.from_numpy(data.y)
    model = Expert()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    generator = torch.Generator().manual_seed(fit.seed)
    checkpoint = folder / 'checkpoint.pt'
    first, trace = 0, []
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=False)
        assert saved['signature'] == signature
        model.load_state_dict(saved['state'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator'])
        torch.set_rng_state(saved['rng'])
        first, trace = saved['epoch'], saved['trace']
    started = time.monotonic()
    for epoch in range(first, 20):
        model.train()
        losses = []
        for chunk in torch.randperm(len(fit.train), generator=generator).split(64):
            ids = fit.train[chunk.numpy()]
            loss = model.objective((x[ids], numeric[ids]), target[ids], fit.mtp)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        trace.append({'epoch': epoch + 1, 'MSE_aux': float(np.mean(losses))})
        tensor_checkpoint(checkpoint, {'state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'generator': generator.get_state(), 'rng': torch.get_rng_state(), 'epoch': epoch + 1, 'trace': trace, 'signature': signature})
    model.eval()
    with torch.inference_mode():
        expert = np.concatenate([model.predict(x[ids], numeric[ids]).numpy() for ids in np.array_split(fit.predict, max(1, (len(fit.predict) + 255) // 256))])
        replay_ids = fit.predict[:min(64, len(fit.predict))]
        replay = model.predict(x[replay_ids], numeric[replay_ids]).numpy()
    assert np.allclose(expert[:len(replay)], replay, atol=1e-7, rtol=0)
    distance = x[fit.predict, :, :32].square().mean((1, 2)).sqrt().numpy()
    observed = np.column_stack((numeric[fit.predict, 2:].numpy(), distance, np.abs(numeric[fit.predict, :2].numpy()).mean(1))).astype(np.float32)
    np.savez(folder / 'predictions.npz', expert=expert, observed=observed)
    save(receipt_path, {'signature': signature, 'trace': trace, 'seconds': time.monotonic() - started,
         'transform_fit_ids': fit.train.tolist(), 'target_inference_input': False, 'replay_maxdiff': float(np.max(np.abs(expert[:len(replay)] - replay))),
         'outputs': {p.name: sha(p) for p in (checkpoint, transform_path, folder / 'predictions.npz')}})
    print(json.dumps({'fit': str(folder), 'epochs': 20, 'seconds': time.monotonic() - started}), flush=True)
    return expert, observed


def oof(data: Data, fit: Fit):
    train = fit.train
    groups = sorted(set(data.groups[train]), key=lambda g: hashlib.sha256(f'20261111:{g}'.encode()).hexdigest())
    assignment = {g: i % 3 for i, g in enumerate(groups)}
    fold_ids = np.array([assignment[g] for g in data.groups[train]])
    save(fit.folder / 'group_split.json', assignment)
    expert, observed = np.zeros((len(train), 4, 2), np.float32), np.zeros((len(train), 6), np.float32)
    for fold in range(3):
        ids = train[fold_ids != fold]
        held = train[fold_ids == fold]
        assert not set(data.groups[ids]) & set(data.groups[held])
        p, f = expert_fit(data, Fit(fit.folder / f'oof{fold}', ids, held, fit.seed, fit.mtp))
        expert[fold_ids == fold], observed[fold_ids == fold] = p, f
    p, f = expert_fit(data, Fit(fit.folder / 'expert', train, fit.predict, fit.seed, fit.mtp))
    return (expert, observed), (p, f)


def combine(config: Config, folder: Path, arrays: tuple, seed: int):
    """고정 OOF 예측에만 gate·C를 fit하고 별도 관측 예측에 적용한다."""
    (expert, hold, observed, target), evaluation = arrays
    inputs = tuple(torch.as_tensor(a, dtype=torch.float32) for a in (expert, hold, observed))
    truth = torch.as_tensor(target, dtype=torch.float32)
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = Combiner(config)
    checkpoint = folder / 'checkpoint.pt'
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=.001, weight_decay=.01) if trainable else None
    first, trace = 0, []
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=False)
        model.load_state_dict(saved['state'])
        if optimizer:
            optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        first, trace = saved['epoch'], saved['trace']
    for epoch in range(first, 100 if trainable else 0):
        prediction = model(inputs)[0]
        loss = (prediction - truth).square().mean() + .1 * ccc_loss(prediction, truth)
        loss = loss + .01 * ((.5 * model.a.tanh()).square().mean() + (.1 * model.b.tanh()).square().mean())
        assert optimizer is not None and torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        trace.append(float(loss.detach()))
        tensor_checkpoint(checkpoint, {'state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state(), 'epoch': epoch + 1, 'trace': trace})
    if not trainable:
        tensor_checkpoint(checkpoint, {'state': model.state_dict(), 'optimizer': None, 'rng': torch.get_rng_state(), 'epoch': 0, 'trace': []})
    model.eval()
    with torch.inference_mode():
        prediction, gate = model(tuple(torch.as_tensor(a, dtype=torch.float32) for a in evaluation))
    save(folder / 'receipt.json', {'checkpoint_sha256': sha(checkpoint), 'seed': seed, 'epochs': len(trace), 'trace': trace,
         'train_rows': len(expert), 'gate_quantiles': np.quantile(gate.numpy(), [0, .05, .5, .95, 1]).tolist(),
         'a': (1 + .5 * model.a.tanh()).detach().tolist(), 'b': (.1 * model.b.tanh()).detach().tolist(), 'OOF_only': True})
    return prediction.numpy(), gate.numpy()
