# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: python entry.py status|predict|replay|evaluate|verify
"""최신 선택 관계 head의 재현과 공식 평가 진입점."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from latest_inference import ROOT, DATA, LATEST, SELECTED, predict, digest

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'baseline/b1/vendor'))
from bundle_paths import RESULTS
from scoring import capture, statistics, merge, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('status','predict','replay','evaluate','verify'))
    parser.add_argument('--sample')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit은 양수여야 합니다')
    torch.set_num_threads(1)
    out = (args.output or RESULTS / 'b1/latest' / f'{args.action}_{time.time_ns()}').resolve()
    if not out.is_relative_to(RESULTS):
        parser.error('출력은 로컬 results 폴더 안이어야 합니다')
    out.mkdir(parents=True, exist_ok=False)
    result = {'methods': SELECTED, 'action': args.action, 'paid_api_calls': 0, 'training_calls': 0}
    if args.action == 'status':
        result.update(condition='관측 GT 객체, 미래 annotation schedule, IoU=0', selected_before_test=True)
    else:
        rows = [json.loads(line) for line in (DATA/'samples.jsonl').read_text().splitlines()]
        rows = [r for r in rows if r['future_count'] and (not args.sample or r['sample_id']==args.sample)]
        if args.action == 'verify':
            ids = {r['sample_id'] for r in json.loads((LATEST/'demo_selection.json').read_text())}
            rows = [r for r in rows if r['sample_id'] in ids]
            for asset in json.loads((LATEST/'provenance.json').read_text()):
                assert digest(LATEST/asset['destination']) == asset['sha256']
        if args.action == 'predict' and not args.sample:
            rows = rows[:1]
        if args.limit:
            rows = rows[:args.limit]
        if not rows:
            raise KeyError(args.sample)
        truths = {}
        if args.action == 'evaluate':
            for mode, cohort in (('sgcls','filtered_small'),('predcls','unfiltered')):
                truths[mode] = {r['case_id']: r['frames'] for r in map(json.loads,(DATA/f'benchmark/protocol/{cohort}/future_gt.jsonl').read_text().splitlines())}
        totals, checks = {}, []
        for row in rows:
            baseline, ours, probabilities = predict(row['sample_id'])
            stem = row['sample_id'].replace(':','__')
            torch.save({'baseline':baseline,'ours':ours,'probabilities':probabilities}, out/f'{stem}.pt')
            if args.action == 'verify':
                reference = torch.load(LATEST/'reference'/f'{stem}.pt', map_location='cpu', weights_only=False)
                diff = float((probabilities-reference['probabilities'][SELECTED[row['mode']]]).abs().max())
                assert diff <= 1e-7, (row['sample_id'], diff)
                checks.append({'sample_id':row['sample_id'],'max_abs_difference':diff})
            if args.action == 'evaluate':
                truth = truths[row['mode']][row['sample_id'].split(':',1)[1]]
                for name, prediction in (('baseline',baseline),('ours',ours)):
                    scored = capture(copy.deepcopy(prediction),truth,row['mode'],np.full(len(prediction['boxes']),-1))
                    for constraint, frames in scored.items():
                        key = row['mode'],row['ratio'],name,constraint
                        stats = statistics(frames)
                        totals[key] = merge([totals[key],stats]) if key in totals else stats
        result.update(processed=len(rows),checks=checks,head_forward=True)
        if totals:
            (out/'metrics.json').write_text(json.dumps([dict(mode=k[0],ratio=k[1],method=k[2],constraint=k[3],metrics=metrics(v)) for k,v in totals.items()],indent=2))
    (out/'receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'output':str(out),**result}))
