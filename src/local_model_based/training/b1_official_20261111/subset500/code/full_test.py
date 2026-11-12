# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
# 실행: 기존 AG Python -B code/full_test.py
"""동결 local head를 기존 전체 test 관측 특징에 적용하고 공식 평가기로 채점한다."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import copy
import csv
import hashlib
import json
import os
import socket
import sys
import time
import numpy as np
import torch
from prepare_full import HERE
from forecast_adapter import forecast, reference_forecast
from compact_graph import observed_bank

ORIGINAL = Path('/mnt/RG2/paper_experiments_20261109/combinations')
DATA = Path('/mnt/RG2/HWP_CJ4DPLEX_AI/src/local_model_based/data/b1')
sys.path.insert(0, '/mnt/RG2/paper_experiments_20261109/b1/legacy')
sys.path.insert(0, '/mnt/RG2/my_works')
from scoring import capture, statistics, merge, metrics
from forecasting.ag_vlm.evaluate import JsonlIndex

NEW = tuple(json.loads((HERE / 'runplan.json').read_text())['new_methods'])
OLD = ('gt_transfer','predicted_direct','unanchored_restore_aux','identity_anchor_aux','identity_anchor_no_aux')
METHODS = ('impartail_ode', 'encoder_hold', *OLD, 'joint_current_hold', *NEW)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deny_network(*args, **kwargs) -> None:
    raise PermissionError('Local-only evaluation')


@lru_cache(maxsize=4)
def index(mode: str, name: str):
    cohort = 'filtered_small' if mode == 'sgcls' else 'unfiltered'
    return JsonlIndex(DATA / f'benchmark/protocol/{cohort}/{name}.jsonl', 'case_id')


@lru_cache(maxsize=22)
def bundle(mode: str, method: str):
    path = (HERE if method in NEW else ORIGINAL) / f'assets/final_{mode}_{method}.pt'
    return torch.load(path, map_location='cpu', weights_only=False)


def predictions(mode: str, graph, future: list[int], official):
    """인수에 미래 정답이 없고 호출 중 모든 파일 읽기를 차단한다."""
    bank = observed_bank(graph, future)
    loaded = {name: bundle(mode, name) for name in (*OLD, *NEW)}
    active = [True]
    def guard(event, args):
        if active[0] and event == 'open':
            raise PermissionError('Prediction cannot read files or future GT')
    sys.addaudithook(guard)
    outputs = {}
    try:
        for name in METHODS:
            if name in ('impartail_ode', 'encoder_hold'):
                output = reference_forecast(graph, future, official if name == 'impartail_ode' else None)
            else:
                selected = 'identity_anchor_aux' if name == 'joint_current_hold' else name
                output = forecast(graph, future, dict(loaded[selected], inference_ablation=name))
            assert all(torch.equal(output[key], value) for key, value in bank.items())
            assert all(torch.isfinite(value).all() for value in output.values())
            outputs[name] = output
    finally:
        active[0] = False
    return outputs


def score_case(row):
    torch.set_num_threads(1)
    socket.socket.connect = deny_network
    sid = row['sample_id']
    stem = sid.replace(':','__')
    target = HERE / f'results/test_records/{stem}.json'
    if target.exists():
        return json.loads(target.read_text())
    receipt_path = DATA / f'impartail/receipts/{stem}.json'
    if not receipt_path.exists():
        result = dict(row, status='missing_observation_features', methods={})
        target.write_text(json.dumps(result))
        return result
    receipt = json.loads(receipt_path.read_text())
    graph_path = DATA / 'impartail' / receipt['compact_path']
    pred_path = DATA / 'impartail' / receipt['prediction_path']
    assert digest(graph_path) == receipt['compact_sha256']
    assert digest(pred_path) == receipt['prediction_sha256']
    graph = torch.load(graph_path, weights_only=False)
    official = torch.load(pred_path, weights_only=False)
    mode, case = sid.split(':',1)
    future = [int(Path(n).stem) for n in index(mode,'cases').get(case)['future_frame_ids']]
    started = time.monotonic()
    outputs = predictions(mode, graph, future, official)
    saved = HERE / f'results/test_predictions/{stem}.pt'
    tmp = saved.with_suffix('.tmp')
    torch.save(outputs, tmp)
    tmp.replace(saved)
    # 예측 파일이 완성된 뒤에만 해당 사례의 정답을 읽는다.
    truth = index(mode,'future_gt').get(case)['frames']
    values = {}
    for name, output in outputs.items():
        captured = capture(copy.deepcopy(output), truth, mode, np.full(len(output['boxes']), -1))
        values[name] = {c:statistics(frames) for c,frames in captured.items()}
    result = dict(row, status='success', methods=values, seconds=time.monotonic()-started,
                  prediction_sha256=digest(saved), future_GT_blocked_during_prediction=True)
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(result))
    tmp.replace(target)
    return result


def main() -> None:
    assert (HERE / 'protocol/selection.json').exists()
    for folder in ('test_records','test_predictions'):
        (HERE / 'results' / folder).mkdir(exist_ok=True)
    rows = [json.loads(line) for line in (DATA/'samples.jsonl').read_text().splitlines()]
    selected = [r for r in rows if r['future_count']]
    frozen = {'runplan_sha256':digest(HERE/'runplan.json'), 'selection_sha256':digest(HERE/'protocol/selection.json'),
              'models':{f'{m}:{n}':digest((HERE if n in NEW else ORIGINAL)/f'assets/final_{m}_{n}.pt') for m in ('sgcls','predcls') for n in (*OLD,*NEW)},
              'code':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}, 'scheduled':len(rows), 'eligible':len(selected), 'paid_API_calls':0}
    frozen_path = HERE/'protocol/test_frozen.json'
    if frozen_path.exists():
        assert json.loads(frozen_path.read_text()) == frozen
    else:
        frozen_path.write_text(json.dumps(frozen,indent=2))
    totals, counts = {}, {}
    missing = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(score_case, selected, chunksize=1),1):
            if result['status'] != 'success':
                missing.append(result['sample_id'])
            for name, constraints in result['methods'].items():
                for constraint, stats in constraints.items():
                    key=(result['mode'], result['ratio'], name, constraint)
                    totals[key] = merge([totals[key],stats]) if key in totals else stats
                    counts[key] = counts.get(key,0)+1
            if i % 100 == 0:
                print(json.dumps({'processed':i,'eligible':len(selected),'missing':len(missing)}),flush=True)
    table=[]
    for (mode,ratio,method,constraint), stats in sorted(totals.items()):
        table.append(dict(mode=mode,ratio=ratio,method=method,constraint=constraint,cases=counts[mode,ratio,method,constraint],
                          **{f'{metric}@{k}':value for k,values in metrics(stats).items() for metric,value in values.items()}))
    (HERE/'results/test_summary.json').write_text(json.dumps(table,indent=2))
    with (HERE/'results/test_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    final={'scheduled':len(rows),'eligible':len(selected),'empty_future':len(rows)-len(selected),'processed':len(selected),'missing':missing,'methods':METHODS,'metric_rows':len(table)}
    (HERE/'results/test_receipt.json').write_text(json.dumps(final,indent=2))
    print(json.dumps(final),flush=True)


if __name__ == '__main__':
    main()
