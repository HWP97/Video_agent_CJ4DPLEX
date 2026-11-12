# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
"""Bounded official-train development; no AG test scores or test labels are loaded."""
from pathlib import Path
import copy
import hashlib
import json
import socket
import sys
import time
import numpy as np
import torch
from application_types import Bundle
from compact_graph import observed_bank
from forecast_adapter import forecast, predict_relations, reference_forecast
from relation_model import RelationDynamics, focal
from supervision import targets
from prepare_full import HERE
B1 = Path('/mnt/RG2/paper_experiments_20261109/b1')
from local_extensions import memory_prior, transform, ranking_loss
sys.path.insert(0, str(B1 / 'legacy'))
from scoring import capture, statistics, merge, metrics

VARIANTS = {'gt_transfer': (False, False, 0., 'gt_history'), 'predicted_direct': (False, False, 0., 'predicted_history'),
            'unanchored_restore_aux': (True, False, 1., 'predicted_history'),
            'identity_anchor_aux': (True, True, 1., 'predicted_history'),
            'identity_anchor_no_aux': (True, True, 0., 'predicted_history')}
VARIANTS.update({'local_temporal': (False, False, 1., 'predicted_history'),
                 'local_memory': (True, True, 1., 'predicted_history'),
                 'local_masked': (True, True, 1., 'predicted_history'),
                 'local_mtp': (True, True, 1., 'predicted_history'),
                 'local_rank': (True, True, 1., 'predicted_history')})
VARIANTS['local_macro'] = (True, True, 1., 'predicted_history')
ORIGINAL = Path('/mnt/RG2/paper_experiments_20261109/combinations')
SEED = 20261111


def deny_network(*args, **kwargs) -> None:
    raise PermissionError('Local training/evaluation forbids network')


def initialize(residual: bool, anchor: bool = False) -> RelationDynamics:
    torch.manual_seed(SEED)
    model = RelationDynamics(residual, anchor)
    if anchor:
        nn = torch.nn
        for head in (model.current, model.future):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
    elif residual:
        with torch.no_grad():
            model.current.weight.add_(model.future.weight)
            model.current.bias.add_(model.future.bias)
    return model


def load_case(row: dict) -> tuple[dict, dict, dict]:
    compact = torch.load(HERE / 'assets/compact_train' / f'{row["mode"]}_{row["video"]}.pt', weights_only=False)
    record = json.loads((HERE / 'assets/supervision' / f'{row["mode"]}_{row["video"]}.json').read_text())
    return compact, record, targets(compact['graph'], record)


def train(rows: list, mode: str, stage: str) -> dict[str, Bundle]:
    arrays = {k: [] for k in ('predicted_history', 'gt_history', 'current', 'future', 'delta', 'base_logits')}
    coverage = []
    middle_targets, middle_deltas = [], []
    for row in rows:
        _, _, data = load_case(row)
        for key in arrays:
            arrays[key].append(data[key])
        coverage.append(dict(sample_id=row['sample_id'], **data['coverage']))
        candidates = data['coverage']['candidates']
        frame_count = len(data['delta']) // candidates
        middle = torch.arange(frame_count) // 2
        middle_targets.append(data['future'].reshape(frame_count, candidates, 26)[middle].flatten(0, 1))
        middle_deltas.append(data['delta'].reshape(frame_count, candidates)[middle].flatten())
    packed = {key: torch.cat(values) for key, values in arrays.items()}
    packed['middle_target'] = torch.cat(middle_targets)
    packed['middle_delta'] = torch.cat(middle_deltas)
    prior = memory_prior(packed['predicted_history'], packed['current'])
    weights = packed['future'].sum(0).clamp_min(1).rsqrt()
    weights = weights / weights.mean()
    (HERE / f'results/{stage}_{mode}_training_coverage.json').write_text(json.dumps(coverage, indent=2))
    bundles, receipts = {}, []
    for name, (residual, anchor, auxiliary, history_key) in VARIANTS.items():
        if not name.startswith('local_'):
            bundles[name] = torch.load(ORIGINAL / 'assets' / f'{stage}_{mode}_{name}.pt', weights_only=False)
            continue
        saved = HERE / 'assets' / f'{stage}_{mode}_{name}.pt'
        receipt_path = HERE / 'results' / f'{stage}_{mode}_{name}_training.json'
        if saved.exists() and receipt_path.exists():
            bundles[name] = torch.load(saved, weights_only=False)
            receipts.append(json.loads(receipt_path.read_text()))
            continue
        model = initialize(residual, anchor)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        started = time.monotonic()
        for epoch in range(8):
            total_loss = 0.
            for indices in torch.randperm(len(packed['delta'])).split(256):
                history = transform(packed[history_key][indices], name, prior, training=True)
                future, current = model(history, packed['delta'][indices], torch.randn(8, len(indices), 64), packed['base_logits'][indices])
                loss = focal(future, packed['future'][indices]) + auxiliary * focal(current, packed['current'][indices])
                if name == 'local_macro':
                    loss = (torch.nn.functional.binary_cross_entropy_with_logits(future, packed['future'][indices], reduction='none') * weights).mean()
                    loss = loss + (torch.nn.functional.binary_cross_entropy_with_logits(current, packed['current'][indices], reduction='none') * weights).mean()
                if name == 'local_rank':
                    loss = loss + .1 * ranking_loss(future, packed['future'][indices])
                if name == 'local_mtp':
                    middle, _ = model(history, packed['middle_delta'][indices], torch.randn(8, len(indices), 64), packed['base_logits'][indices])
                    loss = loss + focal(middle, packed['middle_target'][indices])
                assert torch.isfinite(loss), (stage, mode, name, epoch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                total_loss += loss.item() * len(indices)
            print(json.dumps({'stage': stage, 'mode': mode, 'method': name, 'epoch': epoch + 1,
                              'loss': total_loss / len(packed['delta']), 'seconds': time.monotonic() - started}), flush=True)
        bundle = {'state_dict': model.state_dict(), 'residual': residual, 'anchor': anchor, 'method': name, 'mode': mode,
                  'seed': SEED, 'time_origin': 'actual_relation_cutoff', 'time_scale': 30., 'schema_version': 1, 'epochs': 8, 'training_groups': len(rows),
                  'protocol_sha256': hashlib.sha256((HERE / 'runplan.json').read_bytes()).hexdigest(),
                  'encoder_checkpoint_sha256': torch.load(HERE / 'assets/compact_train' / f'{rows[0]["mode"]}_{rows[0]["video"]}.pt', weights_only=False)['graph']['provenance']['encoder_checkpoint_sha256']}
        if name == 'local_memory':
            bundle['memory_prior'] = prior
        torch.save(bundle, saved)
        bundles[name] = bundle
        receipts.append({'method': name, 'mode': mode, 'stage': stage, 'seconds': time.monotonic() - started,
                         'parameters': sum(p.numel() for p in model.parameters()), 'training_groups': len(rows),
                         'examples': len(packed['delta']), 'epochs': 8, 'last_loss': total_loss / len(packed['delta']), 'device': 'cpu', 'peak_vram_bytes': 0,
                         'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}})
        receipt_path.write_text(json.dumps(receipts[-1], indent=2))
    (HERE / f'results/{stage}_{mode}_training.json').write_text(json.dumps(receipts, indent=2))
    return bundles


def evaluate(rows: list, bundles: dict, mode: str) -> list:
    result = []
    output = HERE / 'results/development_raw'
    output.mkdir(exist_ok=True)
    for row in rows:
        compact = torch.load(HERE / 'assets/compact_train' / f'{mode}_{row["video"]}.pt', weights_only=False)
        graph = compact['graph']
        future_frames = [int(Path(name).stem) for name in row['future_frame_ids']]
        bank = observed_bank(graph, future_frames)
        methods = ['impartail_ode', 'encoder_hold', *VARIANTS, 'joint_current_hold']
        predictions, current_outputs, timings = {}, {}, {}
        for name in methods:
            started = time.monotonic()
            if name in {'impartail_ode', 'encoder_hold'}:
                prediction = reference_forecast(graph, future_frames, compact['official_prediction'] if name == 'impartail_ode' else None)
                from forecast_adapter import observed_logits
                current_outputs[name] = observed_logits(graph)
            else:
                bundle = dict(bundles['identity_anchor_aux' if name == 'joint_current_hold' else name], inference_ablation=name)
                prediction = forecast(graph, future_frames, bundle)
                _, current, _ = predict_relations(graph, future_frames, bundle, current_hold=name == 'joint_current_hold')
                current_outputs[name] = current
            for key in ('boxes', 'labels', 'pred_labels', 'scores', 'pred_scores', 'pair_idx', 'im_idx'):
                assert torch.equal(bank[key], prediction[key]), (name, key)
            torch.save(prediction, output / f'{mode}_{row["video"]}_{name}.pt')
            predictions[name] = prediction
            timings[name] = time.monotonic() - started
        # All predictions have been saved before this process reads the current video's supervised targets.
        record = json.loads((HERE / 'assets/supervision' / f'{mode}_{row["video"]}.json').read_text())
        data = targets(graph, record)
        for name in methods:
            current_truth = data['current'][:data['coverage']['candidates']]
            current = current_outputs[name][:len(current_truth)]
            predicted, correct = current.sigmoid() >= .5, current_truth.bool()
            current_stats = {'attention_correct': int((current[:, :3].argmax(-1) == current_truth[:, :3].argmax(-1)).sum()),
                             'pairs': len(current), 'tp': int((predicted[:, 3:] & correct[:, 3:]).sum()),
                             'fp': int((predicted[:, 3:] & ~correct[:, 3:]).sum()),
                             'fn': int((~predicted[:, 3:] & correct[:, 3:]).sum())}
            prediction = predictions[name]
            scored = capture(copy.deepcopy(prediction), record['future_GT'], mode, np.full(len(prediction['boxes']), -1))
            result.append({'sample_id': row['sample_id'], 'mode': mode, 'method': name, 'video': row['video'],
                           'seconds': timings[name], 'coverage': data['coverage'], 'current_metrics': current_stats,
                           'stats': {c: statistics(frames) for c, frames in scored.items()}})
        print(json.dumps({'development_scored': row['sample_id'], 'methods': len(methods)}), flush=True)
    (HERE / f'results/development_{mode}.json').write_text(json.dumps(result))
    return result


def select_candidate(table: list) -> tuple[str, bool]:
    """Keep the adoption guard separate from mandatory negative diagnostic selection."""
    reference = next(r['metrics']['20'] for r in table if r['method'] == 'impartail_ode')
    preference = [name for name in VARIANTS if name.startswith('local_')]
    candidates = [r for r in table if r['method'] in preference]
    eligible = [r for r in candidates if r['metrics']['20']['R'] + 1e-12 >= reference['R'] and r['metrics']['20']['mR'] > reference['mR']]
    ranked = eligible if eligible else candidates
    chosen = sorted(ranked, key=lambda r: (-r['metrics']['20']['mR'], preference.index(r['method'])))[0]['method']
    return chosen, bool(eligible)


def main() -> None:
    socket.socket.connect = deny_network
    torch.set_num_threads(1)
    manifest = json.loads((HERE / 'protocol/train_manifest.json').read_text())
    assert all((HERE / 'assets/compact_train' / f'{r["mode"]}_{r["video"]}.pt').is_file() for r in manifest if r['status'] == 'ready'), 'Subset train features are not ready'
    selections, tables = {}, []
    for mode in ('sgcls', 'predcls'):
        rows = [r for r in manifest if r['mode'] == mode and r['status'] == 'ready']
        training = [r for r in rows if r['split'] == 'training']
        development = [r for r in rows if r['split'] == 'development']
        assert len(development) == 100
        bundles = train(training, mode, 'development')
        stored = HERE / f'results/development_{mode}.json'
        results = json.loads(stored.read_text()) if stored.exists() else evaluate(development, bundles, mode)
        table = []
        for method in ['impartail_ode', 'encoder_hold', *VARIANTS, 'joint_current_hold']:
            selected = [r for r in results if r['method'] == method]
            stats = merge([r['stats']['no'] for r in selected])
            table.append({'mode': mode, 'method': method, 'groups': len(selected), 'frames': stats['frames'], 'metrics': metrics(stats)})
        reference = table[0]['metrics']['20']
        chosen, qualified = select_candidate(table)
        selections[mode] = {'method': chosen, 'adopted': qualified,
                            'execution_purpose': 'qualified_candidate' if qualified else 'rejected_diagnostic',
                            'reference': reference, 'table': table}
        tables.extend(table)
    selection = {'frozen_at_unix': time.time(), 'modes': selections, 'test_scores_seen': False,
                 'selection_scope': 'development_qualification_only_not_service_adoption',
                 'protocol_sha256': hashlib.sha256((HERE / 'runplan.json').read_bytes()).hexdigest()}
    (HERE / 'protocol/selection.json').write_text(json.dumps(selection, indent=2))
    (HERE / 'results/development_summary.json').write_text(json.dumps(tables, indent=2))
    for mode in selections:
        rows = [r for r in manifest if r['mode'] == mode and r['status'] == 'ready']
        train(rows, mode, 'final')
    print(json.dumps({'completed': True, 'selection': {m: s['method'] for m, s in selections.items()}}), flush=True)


if __name__ == '__main__':
    main()
