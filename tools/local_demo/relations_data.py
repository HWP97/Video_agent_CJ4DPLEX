# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: relations_demo.py에서 사용
"""최신 로컬 관계 예측과 원 주석을 실제 영상에 연결한다."""
from dataclasses import dataclass
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
LOCAL = PROJECT / 'src/local_model_based'
DATA = LOCAL / 'data/b1'
SOURCE = PROJECT.parent / 'my_works/b1/benchmarks/action_genome_20261106/data'
sys.path[:0] = [str(LOCAL/'ours/b1'), str(LOCAL/'baseline/b1/vendor')]
from latest_inference import predict, schedules
from scoring import capture, official, evaluate

OBJECTS = ['background'] + (DATA/'benchmark/data/annotations/object_classes.txt').read_text().splitlines()
RELATIONS = ['looking at','not looking at','unsure','above','beneath','in front of','behind','on the side of','in','carrying','covered by','drinking from','eating','have on back','holding','leaning on','lying on','not contacting','other relation','sitting on','standing on','touching','twisting','wearing','wiping','writing on']


@dataclass(frozen=True, slots=True)
class Candidate:
    sample_id: str
    mode: str
    video_id: str
    baseline: float
    ours: float
    baseline_wc: float
    ours_wc: float


@dataclass(frozen=True, slots=True)
class Snapshot:
    time: float
    image: Path
    boxes: np.ndarray
    classes: np.ndarray
    baseline: tuple[str, ...]
    ours: tuple[str, ...]
    truth: tuple[str, ...]
    baseline_recall: float
    ours_recall: float
    baseline_matches: tuple[tuple[int, ...], ...]
    ours_matches: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class Scene:
    candidate: Candidate
    cutoff: float
    end: float
    native_pts: np.ndarray
    snapshots: tuple[Snapshot, ...]
    source_video: Path


def label(subject: int, obj: int, predicate: int) -> str:
    return f'{OBJECTS[subject]} > {RELATIONS[predicate]} > {OBJECTS[obj]}'


def ranked_labels(entry) -> tuple[str, ...]:
    scores = entry.objects[entry.pairs].prod(1)[:, None] * entry.relations
    return tuple(label(int(entry.classes[entry.pairs[row,0]]),int(entry.classes[entry.pairs[row,1]]),int(predicate)) for row,predicate in official.argsort_desc(scores)[:20])


def matches(entry, truth, mode: str) -> tuple[tuple[int, ...], ...]:
    metrics = {mode+'_recall':{20:[]},mode+'_mean_recall_collect':{20:[[] for _ in range(26)]}}
    found, _, _ = official.evaluate_from_dict(truth,entry.official(),mode,metrics,method='no',threshold=None,num_rel=26,iou_thresh=0.)
    return tuple(tuple(int(i) for i in group) for group in found[:20])


def load_scene(candidate: Candidate) -> Scene:
    torch.set_num_threads(1)
    baseline, ours, _ = predict(candidate.sample_id)
    mode, case_id = candidate.sample_id.split(':',1)
    schedule = schedules(mode)[case_id]
    timeline = json.loads((SOURCE/'frame_records'/f'{Path(candidate.video_id).stem}.json').read_text())
    times = {r['frame_id']:r['pts_s'] for r in timeline['frames']}
    cohort = 'filtered_small' if mode=='sgcls' else 'unfiltered'
    with (DATA/f'benchmark/protocol/{cohort}/future_gt.jsonl').open() as stream:
        truth = next(row['frames'] for row in map(json.loads,stream) if row['case_id']==case_id)
    base_frames = capture(copy.deepcopy(baseline),truth,mode,np.full(len(baseline['boxes']),-1))['no']
    ours_frames = capture(copy.deepcopy(ours),truth,mode,np.full(len(ours['boxes']),-1))['no']
    snapshots = []
    for frame_id, base, updated in zip(schedule['future_frame_ids'],base_frames,ours_frames,strict=True):
        gt = base.truth
        count = len(gt['gt_relations'])
        bm, om = matches(base.entry,gt,mode),matches(updated.entry,gt,mode)
        assert set(i for group in bm for i in group)==set(base.hit_sets[1])
        assert set(i for group in om for i in group)==set(updated.hit_sets[1])
        snapshots.append(Snapshot(float(times[frame_id]),SOURCE/'frames'/frame_id,gt['gt_boxes'],gt['gt_classes'],ranked_labels(base.entry),ranked_labels(updated.entry),tuple(label(int(gt['gt_classes'][s]),int(gt['gt_classes'][o]),int(p)) for s,o,p in gt['gt_relations']),len(base.hit_sets[1])/count,len(updated.hit_sets[1])/count,bm,om))
    np.testing.assert_allclose(np.mean([s.baseline_recall for s in snapshots]),candidate.baseline,atol=1e-10,rtol=0)
    np.testing.assert_allclose(np.mean([s.ours_recall for s in snapshots]),candidate.ours,atol=1e-10,rtol=0)
    return Scene(candidate,float(times[schedule['observed_frame_ids'][-1]]),float(times[schedule['future_frame_ids'][-1]]),np.asarray(timeline['native_frame_pts_s']),tuple(snapshots),SOURCE/'videos'/candidate.video_id)
