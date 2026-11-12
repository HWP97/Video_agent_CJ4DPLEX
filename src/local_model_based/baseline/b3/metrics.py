"""Evaluation-only GT access and pinned collision semantics."""
from collections import defaultdict,namedtuple
import importlib.util
import math
from pathlib import Path
import statistics

PATH=Path(__file__).resolve().with_name('official_metrics.py')
spec=importlib.util.spec_from_file_location('trajnet_pinned_metrics',PATH)
official=importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)
Track=namedtuple('Track','frame pedestrian x y')


def score(case,truth,positions):
    errors=[math.hypot(p[0]-g[0],p[1]-g[1]) if p is not None and g is not None else None
            for p,g in zip(positions or [None]*len(truth['xy']),truth['xy'])]
    values=[v for v in errors if v is not None]
    ade=statistics.mean(values) if values else None
    fde=errors[-1]
    collision=None
    if case['dataset']=='trajnetplusplus' and positions is not None:
        pred=[Track(f,case['obs']['primary_id'],*xy) for f,xy in zip(case['obs']['future_frames'],positions)]
        paths=[[Track(*r) for r in p] for p in truth['raw']['paths']]
        ade=float(official.average_l2(paths[0],pred));fde=float(official.final_l2(paths[0],pred))
        visible={t['pedestrian_id'] for t in case['obs']['tracks']}
        collision=int(any(official.collision(pred,p) for p in paths[1:] if p and p[0].pedestrian in visible))
    diagonal=case['diagonal']
    return {'errors':errors,'ade':ade,'fde':fde,'ade_diagonal':ade/diagonal if ade is not None and diagonal else None,
            'fde_diagonal':fde/diagonal if fde is not None and diagonal else None,
            'collision_gt':collision,'cohort':truth['cohort'],'gt_valid_points':sum(p is not None for p in truth['xy'])}


def mean(values):
    kept=[v for v in values if v is not None]
    return statistics.mean(kept) if kept else None


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        for camera in ('all',r['camera']):
            groups[(r['dataset'],r['partition'],r['reference'],r['method'],r['cohort'],camera)].append(r)
    summaries=[]
    for key,values in sorted(groups.items()):
        by_group=defaultdict(list)
        for r in values: by_group[r['group_id']].append(r)
        summaries.append(dict(zip(('dataset','partition','reference','method','cohort','camera'),key),
            requested=len(values),provided=sum(r['status']=='valid' for r in values),scored=sum(r['ade'] is not None for r in values),
            ade=mean(r['ade'] for r in values) if key[0]=='trajnetplusplus' else None,
            fde=mean(r['fde'] for r in values) if key[0]=='trajnetplusplus' else None,
            sequence_macro_ADE_diagonal=mean(mean(r['ade_diagonal'] for r in v) for v in by_group.values()),
            sequence_macro_FDE_diagonal=mean(mean(r['fde_diagonal'] for r in v) for v in by_group.values()),
            collision_gt_percent=100*mean(r['collision_gt'] for r in values) if any(r['collision_gt'] is not None for r in values) else None,
            note='MOT image pixels are not pooled across resolutions; diagonal macro reported'))
    return summaries
