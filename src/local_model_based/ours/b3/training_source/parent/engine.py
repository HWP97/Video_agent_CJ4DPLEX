# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# How to run: run.py에서 import한다.
"""동일 F 초기 checkpoint와 동일 추가학습예산으로 후보를 비교한다."""
from dataclasses import dataclass
from pathlib import Path
import json
import time
import numpy as np
import torch
from models import Model, Config, features, noise_fit, objectives, collision_loss
from followup_common import sha, save, tensor_checkpoint


@dataclass(frozen=True, slots=True)
class Fit:
    folder: Path
    train: np.ndarray
    validation: np.ndarray
    seed: int
    epochs: int
    initialize: Path | None = None
    final: bool = False


def batch(data, ids: np.ndarray):
    return {k: v if k in ('center', 'spread') else v[ids] for k, v in data.items()}


def predict(model: Model, data, ids: np.ndarray):
    model.eval()
    outputs, gates = [], []
    with torch.inference_mode():
        for chunk in np.array_split(ids, max(1, (len(ids) + 511) // 512)):
            p, g = model(batch(data, chunk))
            outputs.append(torch.stack(p, 1).cpu().numpy())
            gates.append(torch.stack(g, 1).cpu().numpy())
    return np.concatenate(outputs), np.concatenate(gates)


def fit(config: Config, data, specification: Fit):
    s = specification
    folder = s.folder
    folder.mkdir(parents=True, exist_ok=True)
    signature = {'config': config.__dict__ if hasattr(config, '__dict__') else {k: getattr(config, k) for k in config.__slots__},
        'seed': s.seed, 'epochs': s.epochs, 'final': s.final,
        'train_ids_sha256': __import__('hashlib').sha256(s.train.tobytes()).hexdigest(),
        'initial_sha256': sha(s.initialize) if s.initialize else None,
        'sources': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}}
    receipt_path = folder / 'receipt.json'
    checkpoint, resume = folder / 'best.pt', folder / 'resume.pt'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['signature'] == signature
        assert sha(checkpoint) == receipt['checkpoint_sha256']
        return receipt
    torch.manual_seed(s.seed)
    torch.cuda.manual_seed_all(s.seed)
    model = Model(config).cuda()
    if s.initialize:
        original = torch.load(s.initialize, map_location='cuda', weights_only=False)
        model.base.load_state_dict({k.removeprefix('base.'): v for k, v in original['state'].items() if k.startswith('base.')})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.01)
    generator = torch.Generator().manual_seed(s.seed)
    sigma, cap = noise_fit(data, s.train)
    weight, ratio, norm_aux = 0., 0., 0.
    if config.uncertainty:
        probe = batch(data, s.train[np.linspace(0, len(s.train) - 1, min(1024, len(s.train)), dtype=int)])
        p = model(probe)[0]
        primary = objectives(p, probe)[0]
        auxiliary = collision_loss(p, probe, (sigma, config.central))
        params = tuple(p for p in model.parameters() if p.requires_grad)
        main_grad = torch.autograd.grad(primary, params, retain_graph=True, allow_unused=True)
        aux_grad = torch.autograd.grad(auxiliary, params, allow_unused=True)
        norm_main = torch.sqrt(sum(g.square().sum() for g in main_grad if g is not None))
        norm_aux = float(torch.sqrt(sum(g.square().sum() for g in aux_grad if g is not None)))
        if norm_aux > 1e-10:
            weight = min(10., .01 * float(norm_main) / norm_aux)
            ratio = weight * norm_aux / max(float(norm_main), 1e-10)
    first, best, trace = 0, float('inf'), []
    if resume.exists():
        saved = torch.load(resume, map_location='cuda', weights_only=False)
        assert saved['signature'] == signature
        model.load_state_dict(saved['state'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator'].cpu())
        torch.set_rng_state(saved['rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        first, best, trace = saved['epoch'], saved['best'], saved['trace']
    started = time.monotonic()
    for epoch in range(first, s.epochs):
        model.train()
        losses = []
        for chunk in torch.randperm(len(s.train), generator=generator).split(256):
            b = batch(data, s.train[chunk.numpy()])
            p = model(b)[0]
            main, progress = objectives(p, b)
            loss = main + (.1 * progress if config.progress else 0)
            if config.uncertainty:
                loss = loss + weight * min(1, (epoch + 1) / 3) * collision_loss(p, b, (sigma, config.central))
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        values, _ = predict(model, data, s.validation)
        target = data['target'][s.validation].cpu().numpy()
        scales = data['scale'][s.validation].cpu().numpy()
        error = np.linalg.norm(values[:, -1] - target, axis=-1) * scales[:, None]
        row = {'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'ADE': float(error.mean()), 'FDE': float(error[:, -1].mean())}
        trace.append(row)
        state = {'state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'generator': generator.get_state(),
                 'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(), 'epoch': epoch + 1,
                 'signature': signature, 'center': data['center'], 'spread': data['spread'], 'sigma': sigma, 'weight': weight}
        if s.final or row['ADE'] < best:
            best = min(best, row['ADE'])
            tensor_checkpoint(checkpoint, state)
        tensor_checkpoint(resume, dict(state, best=best, trace=trace))
        print(json.dumps({'fit': str(folder), **row}), flush=True)
    saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
    model.load_state_dict(saved['state'])
    values, gates = predict(model, data, s.validation)
    np.savez(folder / 'validation.npz', outputs=values, gates=gates, ids=s.validation)
    target = data['target'][s.validation].cpu().numpy()
    scale = data['scale'][s.validation].cpu().numpy()
    errors = np.linalg.norm(values - target[:, None], axis=-1).mean(2) * scale[:, None]
    prior = np.linalg.norm(data['kalman'][s.validation].cpu().numpy() - target, axis=-1).mean(1) * scale
    selected = {'ADE': float(errors[:, -1].mean()), 'FDE': float((np.linalg.norm(values[:, -1, -1] - target[:, -1], axis=-1) * scale).mean())}
    receipt = {'signature': signature, 'checkpoint_sha256': sha(checkpoint), 'selected_epoch': saved['epoch'],
         'trace': trace, 'metrics': selected, 'validation_prediction_sha256': sha(folder / 'validation.npz'),
         'iteration_ADE': errors.mean(0).tolist(), 'worse_fraction': [(errors[:, 0] > prior).mean().item(), (errors[:, 1] > errors[:, 0]).mean().item()],
         'gate_quantiles': np.quantile(gates, [0, .05, .5, .95, 1]).tolist(), 'noise_sigma': sigma.tolist(), 'noise_cap95': cap.tolist(),
         'noise_fit_train_only': True, 'aux_weight': weight, 'initial_gradient_ratio': ratio, 'aux_gradient_norm': norm_aux,
         'seconds': time.monotonic() - started, 'optimizer_restarted': s.initialize is not None, 'API_calls': 0}
    save(receipt_path, receipt)
    return receipt
