# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# How to run: imported by prepare.py; feature extraction has no GT argument.
"""Conditional features from immutable Sol entries and unique observed geometry."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from runtime import OLD
from adapters import Entry, FAMILIES


@dataclass(frozen=True, slots=True)
class Features:
    values: np.ndarray
    rows: np.ndarray
    predicates: np.ndarray
    baseline: np.ndarray
    matched: int
    candidates: int


def local_relations(request, graph_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Only identical cutoff and unique original-pixel geometry may connect objects."""
    count = len(request['objects'])
    values, valid = np.zeros((count, 26), np.float32), np.zeros(count, np.float32)
    if not graph_path.exists():
        return values, valid
    graph = torch.load(graph_path, map_location='cpu', weights_only=False)
    cutoff = int(Path(request['frames'][-1]['source_frame_id']).stem)
    if graph['actual_relation_cutoff'] != cutoff:
        return values, valid
    probs = torch.cat((graph['attention_logits'].softmax(-1), graph['spatial_probabilities'], graph['contact_probabilities']), -1).numpy()
    last = graph['pair_frames'].max()
    selected = torch.where(graph['pair_frames'] == last)[0].numpy()
    boxes = graph['boxes'][graph['pairs'][selected, 1], 1:].numpy()
    requested = np.asarray([o['bbox_xyxy'] for o in request['objects']])
    for i, box in enumerate(requested):
        hits = np.flatnonzero(np.all(np.isclose(boxes, box, rtol=0, atol=1e-3), axis=1))
        unique_request = np.all(np.isclose(requested, box, rtol=0, atol=1e-3), axis=1).sum() == 1
        if len(hits) == 1 and unique_request:
            values[i], valid[i] = probs[selected[hits[0]]], 1
    return values, valid


def extract(entry: Entry, local: tuple[np.ndarray, np.ndarray], context: tuple[float, int]) -> Features:
    """Each active candidate gets baseline metadata and an optional observed distribution."""
    ratio, future_bin = context
    values, valid = local
    count = len(entry.pairs) // 3
    records, rows, predicates, scores = [], [], [], []
    matched = set()
    for family, (start, length) in enumerate(FAMILIES):
        for row in range(family * count, (family + 1) * count):
            subject, obj = entry.pairs[row]
            object_index = subject if family == 1 else obj
            identity = int(entry.identities[object_index])
            ordered = np.argsort(-entry.relations[row, start:start+length], kind='stable')
            ranks = np.argsort(ordered)
            available = 0 <= identity < len(valid) and bool(valid[identity])
            obs = values[identity] if available else np.zeros(26)
            if available:
                matched.add(identity)
            category = np.eye(37)[int(entry.classes[object_index])]
            for p in range(start, start + length):
                score = float(entry.relations[row, p])
                meta = [score, ranks[p-start]/max(1,length-1), float(np.max(entry.relations[row,start:start+length]))-score,
                        entry.objects[object_index], min(count,10)/10, ratio, future_bin/2]
                records.append(np.r_[meta, category, np.eye(26)[p], obs, float(available)])
                rows.append(row); predicates.append(p); scores.append(score)
    return Features(np.asarray(records, np.float32), np.asarray(rows), np.asarray(predicates),
                    np.asarray(scores, np.float32), len(matched), count)
