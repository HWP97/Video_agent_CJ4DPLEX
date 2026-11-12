# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: runner.py를 통해 호출
"""Sol 캐시 재채점 및 동결 rank_residual 추론."""
import hashlib
import json
import pickle
from pathlib import Path
import time

import numpy as np
import torch
from scoring import evaluate, statistics, merge, metrics
from bundle_paths import RESULTS, REPORTS, artifact_path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/b1'


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def execute(args, output: Path):
    torch.set_num_threads(1)
    rows = [json.loads(line) for line in (DATA / 'samples.jsonl').read_text().splitlines()]
    selected = [row for row in rows if row['future_count'] and (not args.sample or row['sample_id'] == args.sample)]
    if args.limit:
        selected = selected[:args.limit]
    if args.sample and not selected:
        raise KeyError(args.sample)
    summary = {'scheduled':len(rows),'eligible':sum(bool(row['future_count']) for row in rows),'empty_future':sum(not row['future_count'] for row in rows),'method':args.research or 'raw_sol','action':args.action}
    if args.action == 'status':
        return summary
    if args.action == 'verify':
        failures = []
        manifest = REPORTS / 'b1/copy_manifest.jsonl'
        checked = 0
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            target = artifact_path(row['destination'])
            if target.is_symlink() or target.stat().st_nlink != 1 or digest(target) != row['destination_sha256']:
                failures.append(row['destination'])
            checked += 1
        (output / 'verification.json').write_text(json.dumps({'checked':checked,'failures':failures}))
        if failures:
            raise RuntimeError(failures)
        from validate_raw import validate
        return summary | {'checked':checked,'failures':0} | validate(ROOT)
    if args.action == 'replay':
        totals = {}
        count = 0
        for shard in range(8):
            path = RESULTS / f'b1/frozen/results_full_{shard}/per_case.jsonl'
            for line in path.open():
                record = json.loads(line)
                row = record['schedule']
                for name in (('rank_residual',) if args.research else ('sol',)):
                    for constraint, stats in record['methods'][name].items():
                        key = (row['mode'],row['ratio'],name,constraint)
                        totals[key] = merge([totals[key], stats]) if key in totals else stats
                count += 1
        values = [dict(mode=k[0],ratio=k[1],method=k[2],constraint=k[3],metrics=metrics(v)) for k,v in sorted(totals.items())]
        (output / 'metrics.json').write_text(json.dumps(values,indent=2))
        return summary | {'replayed':count,'metric_groups':len(values),'source':'saved_per_case_statistics'}
    models = {}
    if args.research:
        from adaptive import Adaptive, Decoder, predict
        for mode in ('sgcls','predcls'):
            checkpoint = torch.load(DATA / f'models/{mode}_rank_residual.pt',map_location='cpu',weights_only=False)
            model = Adaptive(checkpoint['residual'])
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model.eval()
            models[mode] = Decoder(model,checkpoint['support'])
    checked = 0
    research_checks = 0
    references = {}
    if args.research:
        needed = {row['sample_id'] for row in selected}
        for shard in range(8):
            with (RESULTS / f'b1/frozen/results_full_{shard}/per_case.jsonl').open() as source:
                for line in source:
                    frozen = json.loads(line)
                    identity = frozen['schedule']['sample_id']
                    if identity in needed:
                        references[identity] = frozen['methods']['rank_residual']
            if references.keys() == needed:
                break
        assert references.keys() == needed
    start = time.monotonic()
    with (output / 'per_case.jsonl').open('x') as stream:
        for row in selected:
            path = DATA / 'prepared' / (row['sample_id'].replace(':','__')+'.pkl')
            with path.open('rb') as source:
                record = pickle.load(source)
            result = {}
            for constraint, frames in record['frames'].items():
                scored = []
                for frame, feature in zip(frames,record['features'][constraint]):
                    entry = predict(frame.entry,feature,models[row['mode']]) if args.research else frame.entry
                    assert all(np.array_equal(getattr(entry,k),getattr(frame.entry,k)) for k in ('boxes','classes','objects','pairs','identities'))
                    scored.append(evaluate(entry,frame.truth,row['mode'],constraint))
                result[constraint] = statistics(scored)
                if args.research:
                    assert result[constraint] == references[row['sample_id']][constraint]
                    research_checks += 1
                baseline = statistics([evaluate(f.entry,f.truth,row['mode'],constraint) for f in frames]) if args.research else result[constraint]
                assert baseline == record['baseline_stats'][constraint]
                checked += 1
            stream.write(json.dumps({'sample_id':row['sample_id'],'statistics':result})+'\n')
    return summary | {'processed':len(selected),'baseline_replay_checks':checked,'research_replay_checks':research_checks,'seconds':time.monotonic()-start}
