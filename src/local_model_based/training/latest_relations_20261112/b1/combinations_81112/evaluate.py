"""관측만으로 새 관계 예측을 만들고 기존 공식 평가기로 채점한다."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import copy
import csv
import json
import os
import shutil
import sys
import numpy as np
import torch
from models import CONFIGS, BLENDS, RelationHead, retrieve, blend_baseline
from prepare import ROOT, ORIGINAL, SOURCE, sha
from compact_graph import histories, observed_bank
from sweep_data import probabilities

sys.path.insert(0, '/mnt/RG2/paper_experiments_20261109/b1/legacy')
sys.path.insert(0, '/mnt/RG2/my_works')
from scoring import capture, statistics, merge, metrics
from forecasting.ag_vlm.evaluate import JsonlIndex

DATA = Path('/mnt/RG2/HWP_CJ4DPLEX_AI/src/local_model_based/data/b1')
ACTIVE = [False]


def initialize() -> None:
    torch.set_num_threads(1)
    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto') or (ACTIVE[0] and event == 'open'):
            raise PermissionError('prediction_file_or_network_access_forbidden')
    sys.addaudithook(audit)


@lru_cache(maxsize=4)
def load_models(mode: str, stage: str):
    models = {}
    for config in CONFIGS:
        prefix = mode + '_' + config.name
        path = ROOT / stage / (prefix + '.pt')
        receipt = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == receipt['checkpoint_sha256']
        model = RelationHead(config).eval()
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['signature'] == receipt['signature']
        model.load_state_dict(saved['state'])
        models[config.name] = model
    with np.load(ROOT / 'data' / f'{mode}_memory.npz') as data:
        mask = data['split'] == 'inner_train' if stage == 'inner' else np.ones(len(data['context']), dtype=bool)
        bank = {name: data[name][mask] for name in data.files}
    return models, bank


@lru_cache(maxsize=4)
def index(mode: str, name: str):
    cohort = 'filtered_small' if mode == 'sgcls' else 'unfiltered'
    return JsonlIndex(DATA / f'benchmark/protocol/{cohort}/{name}.jsonl', 'case_id')


def case(task):
    stage, row, signature = task
    sid, mode = row['sample_id'], row['mode']
    stem = sid.replace(':', '__')
    result_path = ROOT / 'evaluation' / stage / (stem + '.json')
    prediction_path = result_path.with_suffix('.pt')
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result['signature'] == signature and sha(prediction_path) == result['prediction_sha256']
        return result
    if stage == 'inner':
        saved = torch.load(SOURCE / 'assets/compact_train' / f'{mode}_{row["video"]}.pt', weights_only=False)
        graph, official = saved['graph'], saved['official_prediction']
        future = [int(Path(name).stem) for name in row['future_frame_ids']]
    else:
        receipt = json.loads((DATA / 'impartail/receipts' / (stem + '.json')).read_text())
        graph_path = DATA / 'impartail' / receipt['compact_path']
        official_path = DATA / 'impartail' / receipt['prediction_path']
        assert sha(graph_path) == receipt['compact_sha256'] and sha(official_path) == receipt['prediction_sha256']
        graph = torch.load(graph_path, weights_only=False)
        official = torch.load(official_path, weights_only=False)
        future = [int(Path(name).stem) for name in index(mode, 'cases').get(sid.split(':', 1)[1])['future_frame_ids']]
    models, memory = load_models(mode, 'inner' if stage == 'inner' else 'full')
    ACTIVE[0] = True
    try:
        h, _ = histories(graph)
        n, c = len(future), len(h)
        history = h.repeat(n, 1, 1)
        scene = h.mean(0, keepdim=True).expand(n * c, -1, -1)
        delta = torch.tensor([(f - graph['actual_relation_cutoff']) / 30 for f in future]).repeat_interleave(c)
        base = probabilities(official)
        bins = torch.bucketize(torch.arange(n), torch.linspace(0, n, 4).round().long()[1:-1], right=True).repeat_interleave(c)
        remembered, confidence = retrieve(history, bins, [row['video']] * (n * c), memory, 'cpu')
        with torch.inference_mode():
            outputs = {name: model(history, scene, delta, base, remembered, confidence) for name, model in models.items()}
        outputs.update({name: blend_baseline(outputs['c04_gate_positive_cap3'], base, fraction) for name, fraction in BLENDS})
        assert all(p.shape == (n * c, 26) and torch.isfinite(p).all() for p in outputs.values())
        bank = observed_bank(graph, future)
    finally:
        ACTIVE[0] = False
    torch.save({'probabilities': outputs, 'bank': bank,
        'baseline': {key: official[key] for key in ('attention_distribution', 'spatial_distribution', 'contacting_distribution')}}, prediction_path)
    truth = json.loads((SOURCE / 'assets/supervision' / f'{mode}_{row["video"]}.json').read_text())['future_GT'] if stage == 'inner' else index(mode, 'future_gt').get(sid.split(':', 1)[1])['frames']
    predictions = {'baseline': dict(bank, **{key: official[key].clone() for key in ('attention_distribution', 'spatial_distribution', 'contacting_distribution')})}
    for name, p in outputs.items():
        predictions[name] = dict(bank, attention_distribution=p[:, :3].clamp_min(1e-7).log(),
            spatial_distribution=p[:, 3:9], contacting_distribution=p[:, 9:])
    values = {}
    for name, prediction in predictions.items():
        scored = capture(copy.deepcopy(prediction), truth, mode, np.full(len(prediction['boxes']), -1))
        values[name] = {constraint: statistics(frames) for constraint, frames in scored.items()}
    result = dict(row, methods=values, signature=signature, prediction_sha256=sha(prediction_path), GT_access_during_prediction=False)
    result_path.write_text(json.dumps(result))
    return result


def main() -> None:
    stage = sys.argv[1]
    assert stage in ('inner', 'test')
    folder = ROOT / 'evaluation' / stage
    folder.mkdir(parents=True, exist_ok=True)
    if stage == 'inner':
        manifest = json.loads((SOURCE / 'protocol/train_manifest.json').read_text())
        validation = {}
        for mode in ('sgcls', 'predcls'):
            with np.load(ROOT / 'data' / f'{mode}.npz') as data:
                validation[mode] = set(data['groups'][data['split'] == 'inner_validation'])
        rows = [r for r in manifest if r['status'] == 'ready' and r['video'] in validation[r['mode']]]
    else:
        assert (ROOT / 'selection_lock.json').exists()
        combined = json.loads((ROOT / 'combined_selection_lock.json').read_text())
        assert not combined['test_scores_read']
        assert all(sha(Path(path)) == value for path, value in combined['inner_summaries'].items())
        rows = [dict(json.loads(line), video=json.loads(line)['video_id']) for line in (DATA / 'samples.jsonl').read_text().splitlines() if json.loads(line)['future_count']]
    snapshot = folder / 'executed_evaluate.py'
    if not snapshot.exists():
        shutil.copy2(Path(__file__), snapshot)
    assert sha(snapshot) == sha(Path(__file__))
    model_stage = 'inner' if stage == 'inner' else 'full'
    contract = {'stage': stage, 'sources': {str(p): sha(p) for p in (Path(__file__), ROOT / 'models.py', ROOT.parent / 'models.py', ROOT.parent.parent / 'b2/models.py')},
        'models': {p.name: sha(p) for p in (ROOT / model_stage).glob('*.json')},
        'data': sha(ROOT / 'data/receipt.json'), 'rows': len(rows)}
    if stage == 'test':
        contract['combined_selection'] = sha(ROOT / 'combined_selection_lock.json')
    contract_path = folder / 'contract.json'
    if contract_path.exists():
        assert json.loads(contract_path.read_text()) == contract
    else:
        contract_path.write_text(json.dumps(contract, indent=2))
    signature = sha(contract_path)
    totals, counts = {}, {}
    with ProcessPoolExecutor(max_workers=4, initializer=initialize) as pool:
        for number, result in enumerate(pool.map(case, [(stage, row, signature) for row in rows]), 1):
            for name, constraints in result['methods'].items():
                for constraint, stats in constraints.items():
                    key = result['mode'], result.get('ratio', .5), name, constraint
                    totals[key] = merge([totals[key], stats]) if key in totals else stats
                    counts[key] = counts.get(key, 0) + 1
            if number % 25 == 0:
                print(json.dumps({'stage': stage, 'processed': number, 'total': len(rows)}), flush=True)
    table = [dict(mode=k[0], ratio=k[1], method=k[2], constraint=k[3], cases=counts[k],
        **{f'{metric}@{cutoff}': v for cutoff, values in metrics(stats).items() for metric, v in values.items()}) for k, stats in sorted(totals.items())]
    (folder / 'summary.json').write_text(json.dumps(table, indent=2))
    with (folder / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    if stage == 'inner':
        selection = {}
        for mode in ('sgcls', 'predcls'):
            subset = [r for r in table if r['mode'] == mode and r['constraint'] == 'no']
            base = next(r for r in subset if r['method'] == 'baseline')
            passing = [r for r in subset if r['R@20'] >= base['R@20'] and r['mR@20'] > base['mR@20']]
            choice = max(passing, key=lambda r: r['mR@20']) if passing else base
            selection[mode] = {'selected': choice['method'], 'baseline': base, 'result': choice}
        (ROOT / 'selection_lock.json').write_text(json.dumps({'selection': selection,
            'inner_summary_sha256': sha(folder / 'summary.json'), 'test_scores_used': False}, indent=2))
    (folder / 'receipt.json').write_text(json.dumps({'stage': stage, 'cases': len(rows), 'methods': len(CONFIGS) + len(BLENDS) + 1,
        'summary_sha256': sha(folder / 'summary.json'), 'contract_sha256': signature, 'missing': 0}, indent=2))


if __name__ == '__main__':
    main()
