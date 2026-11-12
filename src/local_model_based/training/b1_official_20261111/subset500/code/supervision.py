# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
"""Train/dev targets only. This module is never imported by the inference adapter."""
import torch
from application_types import Supervision, TargetAlignmentError
from compact_graph import histories
from forecast_adapter import observed_logits


def relation_target(entity: dict | None) -> torch.Tensor:
    value = torch.zeros(26)
    if entity is not None:
        for key, offset in [('attention_relationship', 0), ('spatial_relationship', 3), ('contacting_relationship', 9)]:
            for index in entity[key]:
                value[int(index) + offset] = 1
    return value


def center_distance(box, reference, scale) -> float:
    b, r = torch.as_tensor(box), torch.as_tensor(reference)
    return float((((b[:2] + b[2:]) - (r[:2] + r[2:])) / (2 * scale[:2])).square().sum())


def choose(frame: list, category: int, reference, scale) -> tuple[dict | None, int]:
    objects = [o for o in frame[1:] if o['class'] == category]
    if not objects:
        return None, 0
    return min(objects, key=lambda o: center_distance(o['bbox'], reference, scale)), int(len(objects) > 1)


def targets(graph: dict, record: dict) -> Supervision:
    predicted, coverage = histories(graph)
    scale = torch.tensor([*graph['image_size_wh'], *graph['image_size_wh']], dtype=torch.float32)
    actual = int(graph['pair_frames'].max())
    observed = record['observation'][:actual + 1]
    gt_histories, currents, futures = [], [], []
    missing, ambiguous = 0, 0
    for pair_index in coverage['last_pair_indices']:
        object_index = graph['pairs'][pair_index, 1]
        box = graph['boxes'][object_index, 1:]
        objects = observed[-1][1:]
        if not objects:
            raise TargetAlignmentError('Official eligible frame has no GT objects')
        aligned = min(objects, key=lambda o: center_distance(o['bbox'], box, scale))
        category, reference = aligned['class'], aligned['bbox']
        sequence = []
        for frame in observed:
            entity, duplicate = choose(frame, category, reference, scale)
            ambiguous += duplicate
            if entity is None:
                continue
            state = torch.zeros(71)
            state[:26] = relation_target(entity)
            state[26 + category] = 1
            state[63:67] = torch.as_tensor(frame[0]['person_bbox']).reshape(-1, 4)[0] / scale
            state[67:71] = torch.as_tensor(entity['bbox']) / scale
            sequence.append(state)
        sequence = sequence[-4:]
        gt_histories.append(torch.stack([sequence[0]] * (4 - len(sequence)) + sequence))
        currents.append(relation_target(aligned))
        future = []
        for frame in record['future_GT']:
            entity, duplicate = choose(frame, category, reference, scale)
            ambiguous += duplicate
            missing += entity is None
            future.append(relation_target(entity))
        futures.append(torch.stack(future))
    future_count = len(record['future_frame_ids'])
    delta = torch.tensor([(int(name.rsplit('/', 1)[-1].split('.')[0]) - graph['actual_relation_cutoff']) / 30 for name in record['future_frame_ids']], dtype=torch.float32)
    return {'predicted_history': predicted.repeat(future_count, 1, 1),
            'gt_history': torch.stack(gt_histories).repeat(future_count, 1, 1),
            'current': torch.stack(currents).repeat(future_count, 1),
            'base_logits': observed_logits(graph).repeat(future_count, 1),
            'future': torch.stack(futures).transpose(0, 1).flatten(0, 1),
            'delta': delta.repeat_interleave(len(currents)),
            'coverage': dict(coverage, future_absent_targets=missing, target_duplicate_choices=ambiguous)}
