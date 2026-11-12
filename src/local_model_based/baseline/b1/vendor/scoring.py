# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
"""Unmodified official evaluator, observed entries, and additive metric statistics."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TypedDict
import copy
import sys
import numpy as np
import torch

from paths import BASE
from adapters import Entry
from contracts import Statistics,Metrics,Complementarity

sys.path.insert(0,str(BASE/'code'))
from eval_ag import evaluators
from eval_replay import restore
from lib.supervised import evaluation_recall as official


class GroundTruth(TypedDict):
    gt_classes:np.ndarray
    gt_boxes:np.ndarray
    gt_relations:np.ndarray


@dataclass(frozen=True,slots=True)
class Frame:
    truth:GroundTruth
    entry:Entry
    hit_sets:tuple[frozenset[int],...]
    top20_predicates:tuple[int,...]


def capture(prediction,truth,mode:str,identities:np.ndarray) -> dict[str,list[Frame]]:
    """Replay WC then NC exactly; capture the effective entries without mutation."""
    captured={'with':[],'no':[]}
    original=official.evaluate_from_dict

    def callback(gt_entry,pred_entry,current_mode,result_dict,**kwargs):
        returned=original(gt_entry,pred_entry,current_mode,result_dict,**kwargs)
        entry=Entry(pred_entry['pred_boxes'],pred_entry['pred_classes'],pred_entry['obj_scores'],
                    pred_entry['pred_rel_inds'],pred_entry['rel_scores'],identities)
        sets=tuple(frozenset(int(i) for group in returned[0][:k] for i in group) for k in (10,20,50))
        captured[kwargs['method']].append(Frame(gt_entry,entry,sets,tuple(int(row[4]) for row in returned[1][:20])))
        return returned

    official.evaluate_from_dict=callback
    relations=list(map(str,range(26)))
    vocab=SimpleNamespace(object_classes=list(map(str,range(37))),relationship_classes=relations,
                          attention_relationships=relations[:3],spatial_relationships=relations[3:9],contacting_relationships=relations[9:])
    try:
        for evaluator in evaluators(vocab,mode).values():
            evaluator.evaluate_scene_graph(restore(copy.deepcopy(truth)),prediction)
    finally:
        official.evaluate_from_dict=original
    return captured


def evaluate(entry:Entry,truth:GroundTruth,mode:str,constraint:str) -> Frame:
    """Score a post-softmax adapter entry with the same official ranking/matching."""
    if not len(entry.pairs):
        return Frame(truth,entry,(frozenset(),)*3,())
    raw={mode+'_recall':{k:[] for k in (10,20,50)},
         mode+'_mean_recall_collect':{k:[[] for _ in range(26)] for k in (10,20,50)}}
    returned=official.evaluate_from_dict(truth,entry.official(),mode,raw,method=constraint,
                                         threshold=None,num_rel=26,iou_thresh=0.0)
    sets=tuple(frozenset(int(i) for group in returned[0][:k] for i in group) for k in (10,20,50))
    return Frame(truth,entry,sets,tuple(int(row[4]) for row in returned[1][:20]))


def empty_statistics() -> Statistics:
    return {'frames':0,'R_sum':[0.,0.,0.],
            'class_sum':np.zeros((3,26)).tolist(),'class_frames':[0]*26,
            'hits':np.zeros((3,26),dtype=int).tolist(),'counts':[0]*26,
            'object_hits':[0]*37,'object_counts':[0]*37,'top20_family':[0]*3}


def statistics(frames:list[Frame]) -> Statistics:
    """Official frame R and per-predicate frame means, with micro diagnostics separate."""
    result=empty_statistics()
    result['frames']=len(frames)
    sums=np.zeros((3,26))
    hits=np.zeros((3,26),dtype=int)
    for frame in frames:
        predicates=frame.truth['gt_relations'][:,2].astype(int)
        counts=np.bincount(predicates,minlength=26)
        result['counts']=(np.asarray(result['counts'])+counts).tolist()
        result['class_frames']=(np.asarray(result['class_frames'])+(counts>0)).tolist()
        for index,matched in enumerate(frame.hit_sets):
            local=np.bincount(predicates[list(matched)],minlength=26)
            hits[index]+=local
            sums[index]+=np.divide(local,counts,out=np.zeros(26),where=counts>0)
            result['R_sum'][index]+=len(matched)/len(predicates)
        for i,(subject,obj,_) in enumerate(frame.truth['gt_relations']):
            subject_class=int(frame.truth['gt_classes'][subject])
            object_class=int(frame.truth['gt_classes'][obj])
            label=object_class if subject_class==1 else subject_class
            result['object_counts'][label]+=1
            result['object_hits'][label]+=i in frame.hit_sets[1]
        for predicate in frame.top20_predicates:
            family=int(predicate>=3)+int(predicate>=9)
            result['top20_family'][family]+=1
    result['class_sum']=sums.tolist()
    result['hits']=hits.tolist()
    return result


def merge(records:list[Statistics]) -> Statistics:
    result=empty_statistics()
    for record in records:
        result['frames']+=record['frames']
        for key in result.keys()-{'frames'}:
            result[key]=(np.asarray(result[key])+np.asarray(record[key])).tolist()
    return result


def metrics(record:Statistics) -> Metrics:
    frames=record['frames']
    if not frames:
        return {str(k):{'R':None,'mR':None} for k in (10,20,50)}
    counts=np.asarray(record['class_frames'])
    return {str(k):{'R':record['R_sum'][i]/frames,
                    'mR':float(np.divide(record['class_sum'][i],counts,out=np.zeros(26),where=counts>0).mean())}
            for i,k in enumerate((10,20,50))}


def complementarity(hold:list[Frame],sol:list[Frame]) -> Complementarity:
    """Development-only union oracle; no candidate selection is returned to inference."""
    result={key:[0]*26 for key in ('both','hold_only','sol_only','neither','count')}
    for h,s in zip(hold,sol):
        assert np.array_equal(h.truth['gt_relations'],s.truth['gt_relations'])
        for i,(_,_,predicate) in enumerate(h.truth['gt_relations']):
            in_h,in_s=i in h.hit_sets[1],i in s.hit_sets[1]
            category='both' if in_h and in_s else 'hold_only' if in_h else 'sol_only' if in_s else 'neither'
            result[category][int(predicate)]+=1
            result['count'][int(predicate)]+=1
    return result
