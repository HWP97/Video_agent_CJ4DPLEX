# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: ours_cli.py evaluate|replay
"""기존 동결 결과와 같은 공식 평가기로 ours를 대조한다."""
import json
import pickle
import time
from pathlib import Path

from adaptive import predict
from bundle_paths import RESULTS
from scoring import evaluate, statistics, merge, metrics
from ours_method import decoder

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/b1'


def score(args, output: Path):
    name = args.research or 'macro_direct'
    rows = [row for row in map(json.loads,(DATA / 'samples.jsonl').open()) if row['future_count'] and (not args.sample or row['sample_id']==args.sample)]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise KeyError(args.sample)
    needed = {row['sample_id'] for row in rows}
    references = {}
    for shard in range(8):
        with (RESULTS / f'b1/frozen/results_full_{shard}/per_case.jsonl').open() as stream:
            for line in stream:
                record = json.loads(line)
                if record['schedule']['sample_id'] in needed:
                    references[record['schedule']['sample_id']] = record['methods'][name]
    assert references.keys() == needed
    models = {mode:decoder(mode,name) for mode in ('sgcls','predcls')} if args.action == 'evaluate' else {}
    totals = {}
    replay_checks = 0
    started = time.monotonic()
    with (output / 'per_case.jsonl').open('x') as stream:
        for row in rows:
            result = references[row['sample_id']]
            if args.action == 'evaluate':
                path = DATA / 'prepared' / (row['sample_id'].replace(':','__')+'.pkl')
                with path.open('rb') as source:
                    record = pickle.load(source)
                result = {}
                for constraint,frames in record['frames'].items():
                    scored = [evaluate(predict(frame.entry,feature,models[row['mode']]),frame.truth,row['mode'],constraint) for frame,feature in zip(frames,record['features'][constraint])]
                    result[constraint] = statistics(scored)
                    assert result[constraint] == references[row['sample_id']][constraint]
                    replay_checks += 1
            stream.write(json.dumps({'sample_id':row['sample_id'],'method':name,'statistics':result})+'\n')
            for constraint,stats in result.items():
                key = (row['mode'],row['ratio'],constraint)
                totals[key] = merge([totals[key],stats]) if key in totals else stats
    values = [dict(mode=key[0],ratio=key[1],constraint=key[2],method=name,metrics=metrics(stats)) for key,stats in sorted(totals.items())]
    (output / 'metrics.json').write_text(json.dumps(values,indent=2))
    return {'scheduled':13948,'eligible':13635,'empty_future':313,'processed':len(rows),'frozen_statistics_checks':replay_checks,'metric_groups':len(values),'source':'fresh_official_evaluation' if args.action=='evaluate' else 'saved_per_case_statistics','seconds':time.monotonic()-started}
