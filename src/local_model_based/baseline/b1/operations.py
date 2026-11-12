# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: entry.py evaluate --limit 1
"""동결 ImparTail 예측을 원 공식 평가기로 채점한다."""
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from scoring import capture, statistics, merge, metrics
from bundle_paths import REPORTS, artifact_path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/b1'


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def execute(args, output: Path):
    torch.set_num_threads(1)
    rows = [json.loads(line) for line in (DATA / 'samples.jsonl').read_text().splitlines()]
    summary = {'scheduled':len(rows),'eligible':sum(bool(r['future_count']) for r in rows),'empty_future':sum(not r['future_count'] for r in rows),'action':args.action,'condition':'PGAGS 관측 GT bbox; GAGS 관측 GT bbox/class; 미래 annotation frame schedule; IoU=0'}
    if args.action == 'status':
        return summary
    if args.action == 'verify':
        checked = 0
        failures = []
        for line in (REPORTS / 'b1/copy_manifest.jsonl').read_text().splitlines():
            row = json.loads(line)
            path = artifact_path(row['destination'])
            if path.is_symlink() or path.stat().st_nlink != 1 or digest(path) != row['destination_sha256']:
                failures.append(row['destination'])
            checked += 1
        (output / 'verification.json').write_text(json.dumps({'checked':checked,'failures':failures}))
        if failures:
            raise RuntimeError(failures)
        from check_receipts import verify
        return summary | {'checked':checked,'failures':0} | verify(ROOT)
    selected = [r for r in rows if r['future_count'] and (not args.sample or args.sample == r['sample_id'])]
    if args.limit:
        selected = selected[:args.limit]
    if args.sample and not selected:
        raise KeyError(args.sample)
    truths = {}
    for mode,cohort in (('sgcls','filtered_small'),('predcls','unfiltered')):
        truths[mode] = {r['case_id']:r['frames'] for r in map(json.loads,(DATA / f'benchmark/protocol/{cohort}/future_gt.jsonl').open())}
    totals = {}
    started = time.monotonic()
    with (output / 'per_case.jsonl').open('x') as stream:
        for row in selected:
            stem = row['sample_id'].replace(':','__')
            receipt = json.loads((DATA / f'impartail/receipts/{stem}.json').read_text())
            path = DATA / 'impartail' / receipt['prediction_path']
            assert digest(path) == receipt['prediction_sha256']
            prediction = torch.load(path,map_location='cpu',weights_only=False)
            truth = truths[row['mode']][row['sample_id'].split(':',1)[1]]
            scored = capture(copy.deepcopy(prediction),truth,row['mode'],np.full(len(prediction['boxes']),-1))
            values = {constraint:statistics(frames) for constraint,frames in scored.items()}
            stream.write(json.dumps({'sample_id':row['sample_id'],'statistics':values})+'\n')
            for constraint, stats in values.items():
                key = (row['mode'],row['ratio'],constraint)
                totals[key] = merge([totals[key],stats]) if key in totals else stats
    values = [dict(mode=k[0],ratio=k[1],constraint=k[2],metrics=metrics(v)) for k,v in sorted(totals.items())]
    (output / 'metrics.json').write_text(json.dumps(values,indent=2))
    return summary | {'processed':len(selected),'failures':0,'seconds':time.monotonic()-started,'official_evaluator_sha256':digest(ROOT / 'baseline/b1/vendor/benchmark/model/SceneSayer/lib/supervised/evaluation_recall.py')}
