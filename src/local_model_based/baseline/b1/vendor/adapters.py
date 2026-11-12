# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Observation-only score adapters. No GT, file, provider, or model access."""
from __future__ import annotations

from dataclasses import dataclass,replace
from typing import Final
import numpy as np
from numpy.typing import NDArray

FAMILIES: Final=((0,3),(3,6),(9,17))
Key=tuple[int,int,int]


@dataclass(frozen=True,slots=True)
class AdapterError(ValueError):
    reason:str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True,slots=True)
class Entry:
    boxes:NDArray
    classes:NDArray
    objects:NDArray
    pairs:NDArray
    relations:NDArray
    identities:NDArray

    def official(self) -> dict[str,NDArray]:
        return {'pred_boxes':self.boxes,'pred_classes':self.classes,
                'obj_scores':self.objects,'pred_rel_inds':self.pairs,'rel_scores':self.relations}


def validate(entry:Entry) -> None:
    """Reject malformed cached values before rankings or fusion."""
    if len(entry.pairs)%3 or entry.relations.shape!=(len(entry.pairs),26):
        raise AdapterError('invalid_relation_shape')
    if not all(np.isfinite(value).all() for value in (entry.boxes,entry.objects,entry.relations)):
        raise AdapterError('nonfinite_prediction')
    if len(entry.pairs) and (entry.pairs.min()<0 or entry.pairs.max()>=len(entry.classes)):
        raise AdapterError('invalid_pair_index')
    if np.any(entry.classes<1) or np.any(entry.classes>36):
        raise AdapterError('invalid_object_class')


def common_scale(entry:Entry,weights:tuple[float,float]|None,q:NDArray) -> Entry:
    """Preserve each Sol family rank; keep attention and object scores unchanged."""
    validate(entry)
    if weights is None:
        return entry
    if not all(np.isfinite(w) and w>0 for w in weights):
        raise AdapterError('invalid_family_weight')
    scores=entry.relations.copy()
    count=len(scores)//3
    for family,weight in ((1,weights[0]),(2,weights[1])):
        start,length=FAMILIES[family]
        for row in range(family*count,(family+1)*count):
            order=np.argsort(-scores[row,start:start+length],kind='stable')
            values=weight*np.interp(np.arange(length)/(length-1),(0.,0.5,1.),q)
            scores[row,start+order]=values
    return replace(entry,relations=scores)


def restrict(entry:Entry,common:frozenset[int]) -> Entry:
    """Filter physical observation IDs, without joining on predicted classes."""
    count=len(entry.pairs)//3
    keep=[i for i in range(count) if int(entry.identities[entry.pairs[i,1]]) in common]
    indices=np.array([i+family*count for family in range(3) for i in keep],dtype=int)
    return replace(entry,pairs=entry.pairs[indices],relations=entry.relations[indices])


def rank_candidates(entry:Entry) -> dict[Key,int]:
    """Deduplicate only identical physical ID/class/predicate; deterministic ties."""
    validate(entry)
    scores={}
    count=len(entry.pairs)//3
    for family,(start,length) in enumerate(FAMILIES):
        for row in range(family*count,(family+1)*count):
            subject,obj=entry.pairs[row]
            object_index=subject if family==1 else obj
            identity=int(entry.identities[object_index])
            if identity<0:
                raise AdapterError('unconfirmed_object_identity')
            label=int(entry.classes[object_index])
            factor=float(entry.objects[subject]*entry.objects[obj])
            for predicate in range(start,start+length):
                key=(identity,label,predicate)
                score=factor*float(entry.relations[row,predicate])
                scores[key]=max(score,scores.get(key,-np.inf))
    return {key:rank for rank,(key,_) in enumerate(sorted(scores.items(),key=lambda item:(-item[1],item[0])),1)}


def fusion(hold:Entry,sol:Entry,alpha:float) -> Entry:
    """Global reciprocal-rank fusion, including source object scores only in ranks."""
    if not np.isfinite(alpha) or not 0<=alpha<=1:
        raise AdapterError('invalid_fusion_alpha')
    validate(hold)
    validate(sol)
    if alpha==0:
        return hold
    if alpha==1:
        return sol
    ranks_h,ranks_s=rank_candidates(hold),rank_candidates(sol)
    keys=sorted(set(ranks_h)|set(ranks_s))
    hypotheses=sorted({(identity,label) for identity,label,_ in keys})
    count=len(hypotheses)
    boxes=np.full((count+1,4),0.5,dtype=np.float32)
    classes=np.array([1]+[label for _,label in hypotheses],dtype=np.int64)
    ids=np.array([-1]+[identity for identity,_ in hypotheses],dtype=np.int64)
    forward=np.array([[0,i+1] for i in range(count)],dtype=np.int64).reshape(-1,2)
    pairs=np.concatenate((forward,forward[:,::-1],forward))
    scores=np.zeros((3*count,26),dtype=np.float32)
    lookup={key:i for i,key in enumerate(hypotheses)}
    for key in keys:
        identity,label,predicate=key
        value=0.
        if key in ranks_h:
            value+=(1-alpha)/(60+ranks_h[key])
        if key in ranks_s:
            value+=alpha/(60+ranks_s[key])
        family=int(predicate>=3)+int(predicate>=9)
        scores[family*count+lookup[identity,label],predicate]=value
    return Entry(boxes,classes,np.ones(count+1,dtype=np.float32),pairs,scores,ids)


def observation_ids(native_pairs:NDArray,native_box_count:int,observed_boxes:NDArray,
                    vlm_boxes:NDArray,cutoffs_match:bool) -> tuple[NDArray,frozenset[int]]:
    """Restore proven annotation slot order; ambiguous geometry never falls back to class."""
    if not cutoffs_match or native_box_count!=len(observed_boxes):
        raise AdapterError('cutoff_or_slot_count_mismatch')
    if not len(native_pairs) or np.any(native_pairs[:,0]!=0):
        raise AdapterError('unconfirmed_human_slot')
    if np.any(native_pairs<0) or native_pairs.max()>=native_box_count:
        raise AdapterError('invalid_observed_slot')
    ids=np.full(native_box_count,-1,dtype=np.int64)
    for slot in set(int(p[1]) for p in native_pairs):
        matches=np.flatnonzero(np.all(np.isclose(vlm_boxes,observed_boxes[slot],rtol=0,atol=1e-5),axis=1))
        if len(matches)!=1:
            raise AdapterError('nonunique_observed_bbox')
        if int(matches[0]) in ids:
            raise AdapterError('duplicate_physical_id')
        ids[slot]=matches[0]
    return ids,frozenset(int(value) for value in ids if value>=0)
