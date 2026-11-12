# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scipy", "tokenizers"]
# ///
# How to run: CUDA_VISIBLE_DEVICES=0 python -B train.py
"""단일 F 궤적과 epoch0을 보존하는 작은 추가학습을 실행한다."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import torch
from common import (FOLDER, OLD, EPOCHS, SEEDS, Data, Model, CONFIGS, features,
                    objectives, predict, batch, load_data, errors, setup, sha,
                    save, freeze, tensor_checkpoint, ids_sha)


def fit(folder: Path, data: Data, train: np.ndarray, validation: np.ndarray,
        seed: int, epochs: int, parent: Path | None = None,
        moments: bool = False, final: bool = False) -> None:
    """초기 실제 점수에서 선택을 시작하고 지정 epoch의 가중치를 저장한다."""
    folder.mkdir(parents=True, exist_ok=True)
    signature = {'seed': seed, 'epochs': epochs, 'train': ids_sha(train),
                 'validation': ids_sha(validation), 'parent': sha(parent) if parent else None,
                 'moments': moments, 'final': final, 'source': sha(Path(__file__))}
    receipt_path = folder / 'receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['signature'] == signature
        assert all(sha(folder / p) == h for p, h in receipt['checkpoints'].items())
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Model(CONFIGS[0]).cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0001 if parent else .001, weight_decay=.01)
    moment_steps = []
    if parent:
        initial = torch.load(parent, map_location='cuda', weights_only=False)
        assert initial['signature']['train_ids_sha256'] == ids_sha(train)
        model.load_state_dict(initial['state'])
        data['center'], data['spread'] = initial['center'], initial['spread']
        if moments:
            optimizer.load_state_dict(initial['optimizer'])
            for group in optimizer.param_groups:
                group['lr'] = .0001
            for parameter, state in optimizer.state.items():
                assert state['exp_avg'].shape == parameter.shape == state['exp_avg_sq'].shape
                moment_steps.append(float(state['step']))
            assert len(moment_steps) == 10 and min(moment_steps) > 0
    generator = torch.Generator().manual_seed(seed)
    trace, checkpoints = [], {}
    initial_output, initial_ade, initial_fde = errors(model, data, validation)
    best, selected = initial_ade if parent else float('inf'), 0
    started = time.monotonic()
    for epoch in range(epochs + 1):
        if epoch:
            model.train()
            order = torch.randperm(len(train), generator=generator).numpy()
            losses = []
            for start in range(0, len(order), 256):
                chunk = order[start:start + 256]
                values = batch(data, train[chunk])
                loss = objectives(model(values)[0], values)[0]
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            outputs, ade, fde = errors(model, data, validation)
            row = {'epoch': epoch, 'ADE': ade, 'FDE': fde, 'loss': float(np.mean(losses)), 'batch_order_sha256': ids_sha(train[order])}
        else:
            outputs, ade, fde = initial_output, initial_ade, initial_fde
            row = {'epoch': 0, 'ADE': ade, 'FDE': fde, 'loss': None, 'batch_order_sha256': None}
        trace.append(row)
        state = {'state': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'center': data['center'], 'spread': data['spread'], 'epoch': epoch,
                 'generator': generator.get_state(), 'rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state(), 'signature': signature}
        if epoch == 0 or epoch in EPOCHS or (final and epoch == epochs):
            path = folder / f'epoch{epoch}.pt'
            tensor_checkpoint(path, state)
            checkpoints[path.name] = sha(path)
        if (parent and epoch == 0) or (epoch > 0 and (final or ade < best)):
            best, selected = ade, epoch
            tensor_checkpoint(folder / 'best.pt', state)
        if epoch in EPOCHS or (parent and (epoch == 0 or epoch == selected)):
            np.savez(folder / f'validation_epoch{epoch}.npz', outputs=outputs, ids=validation)
        print(json.dumps({'folder': str(folder.relative_to(FOLDER)), **row}), flush=True)
    checkpoints['best.pt'] = sha(folder / 'best.pt')
    save(receipt_path, {'signature': signature, 'trace': trace, 'selected_epoch': selected,
                       'checkpoints': checkpoints, 'initial_ADE': initial_ade,
                       'moment_steps_restored': moment_steps, 'seconds': time.monotonic() - started,
                       'epoch0_candidate': parent is not None, 'API_calls': 0,
                       'Kalman_ADE': float(((data['kalman'][validation] - data['target'][validation]).norm(dim=-1).mean(1) * data['scale'][validation]).mean())})


def inner() -> None:
    """각 fold의 자기 train만으로 선택하고 전체 선택을 동결한다."""
    for fold in (1, 2, 3):
        for split in (0, 1):
            data, train, validation, groups = load_data(fold, split)
            for seed in SEEDS:
                fit(FOLDER / 'inner' / f'fold{fold}' / f'split{split}' / str(seed) / 'F', data, train, validation, seed, 25)
        data, train, validation, groups = load_data(fold, 0)
        parent = OLD / 'inner' / f'fold{fold}' / '20261111/F/best.pt'
        for name, moments in (('B3-D03', False), ('B3-D04', True)):
            fit(FOLDER / 'inner' / f'fold{fold}' / name, data, train, validation, SEEDS[0], 10, parent, moments)
        selections = {}
        for seed in SEEDS:
            candidates = []
            for epoch in EPOCHS:
                ratios, ades = [], []
                for split in (0, 1):
                    receipt = json.loads((FOLDER / 'inner' / f'fold{fold}' / f'split{split}' / str(seed) / 'F/receipt.json').read_text())
                    ade = next(r['ADE'] for r in receipt['trace'] if r['epoch'] == epoch)
                    prior = receipt['Kalman_ADE']
                    ratios.append(ade / prior)
                    ades.append(ade)
                candidates.append({'epoch': epoch, 'ratios': ratios, 'ADE': ades, 'worst_ratio': max(ratios)})
            selections[str(seed)] = {'robust_epoch': min(candidates, key=lambda r: (r['worst_ratio'], r['epoch']))['epoch'],
                                     'single_epoch': min(candidates, key=lambda r: (r['ADE'][0], r['epoch']))['epoch'], 'candidates': candidates}
        additions = {name: json.loads((FOLDER / 'inner' / f'fold{fold}' / name / 'receipt.json').read_text())['selected_epoch'] for name in ('B3-D03', 'B3-D04')}
        freeze(FOLDER / f'selection_fold{fold}.json', {'fold': fold, 'seeds': selections, 'additional_epochs': additions,
               'outer_labels_read': False, 'registration_sha256': sha(FOLDER / 'registration.json')})
    freeze(FOLDER / 'all_folds_frozen.json', {str(f): sha(FOLDER / f'selection_fold{f}.json') for f in (1, 2, 3)})


def full() -> None:
    """동결된 fold별 길이로 전체 train을 재학습한다."""
    frozen = json.loads((FOLDER / 'all_folds_frozen.json').read_text())
    assert all(sha(FOLDER / f'selection_fold{f}.json') == frozen[str(f)] for f in (1, 2, 3))
    for fold in (1, 2, 3):
        lock = json.loads((FOLDER / f'selection_fold{fold}.json').read_text())
        data, train, validation, groups = load_data(fold, None)
        for seed in SEEDS:
            epochs = sorted({lock['seeds'][str(seed)][p + '_epoch'] for p in ('robust', 'single')})
            fit(FOLDER / 'full' / f'fold{fold}' / str(seed) / 'F', data, train, validation, seed, max(epochs), final=True)
        parent = OLD / 'full' / f'fold{fold}' / '20261111/F/best.pt'
        for name, moments in (('B3-D03', False), ('B3-D04', True)):
            fit(FOLDER / 'full' / f'fold{fold}' / name, data, train, validation, SEEDS[0], lock['additional_epochs'][name], parent, moments, True)


if __name__ == '__main__':
    setup()
    freeze(FOLDER / 'training_source_lock.json', {str(p): sha(p) for p in [Path(__file__), FOLDER / 'common.py', OLD / 'models.py', OLD / 'engine.py', FOLDER.parents[1] / 'followup_common.py']})
    inner()
    full()
