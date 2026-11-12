# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
"""Small, observed-only graph contract shared with the full benchmark executor."""
from pathlib import Path
import numpy as np
import torch

SCHEMA_VERSION = 1


def compact_observation(encoded_entry: dict, observed_frame_ids: list[str], mode: str,
                        image_size_wh: tuple[int, int], provenance: dict) -> dict:
    """Call in a dsgdetr forward hook; return CPU tensors, never fmaps/features/GT."""
    if mode not in {'sgcls', 'predcls'}:
        raise ValueError('Unsupported mode')
    required = {'sample_id', 'encoder_checkpoint_sha256', 'encoder_source_sha256', 'observed_image_sha256'}
    if not required <= provenance.keys():
        raise ValueError(f'Missing provenance: {required - provenance.keys()}')
    if len(provenance['observed_image_sha256']) != len(observed_frame_ids):
        raise ValueError('Image hash count differs from observed schedule')
    frames = [int(Path(name).stem) for name in observed_frame_ids]
    if not frames or any(b <= a for a, b in zip(frames, frames[1:])):
        raise ValueError('Observed frames must increase')
    label_key, score_key = ('labels', 'scores') if mode == 'predcls' else ('pred_labels', 'pred_scores')
    mapping = {'boxes': 'boxes', 'pairs': 'pair_idx', 'pair_frames': 'im_idx', 'classes': label_key,
               'object_scores': score_key, 'attention_logits': 'attention_distribution',
               'spatial_probabilities': 'spatial_distribution', 'contact_probabilities': 'contacting_distribution'}
    result = {dest: encoded_entry[src].detach().cpu().clone() for dest, src in mapping.items()}
    if not len(result['pairs']):
        raise ValueError('No observed relation pairs')
    if result['boxes'][:, 0].max() >= len(frames) or result['pair_frames'].max() >= len(frames):
        raise ValueError('Non-observed frame in compact graph')
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError('Nonfinite observed graph')
    distribution = encoded_entry.get('distribution')
    result['class_distribution'] = distribution.detach().cpu().clone() if distribution is not None else None
    result.update(schema_version=SCHEMA_VERSION, mode=mode, frame_indices=frames,
                  observed_frame_ids=list(observed_frame_ids), nominal_cutoff=frames[-1],
                  actual_relation_cutoff=frames[int(result['pair_frames'].max())],
                  image_size_wh=list(image_size_wh), box_coordinates='original_image_pixel_xyxy',
                  class_distribution_semantics='sgcls: probability columns class1..36; predcls: absent or detector-provided, unused',
                  instance_policy='observed predicted class history; duplicate class chooses closest normalized box center; no physical identity guarantee',
                  provenance=dict(provenance))
    return result


def histories(graph: dict) -> tuple[torch.Tensor, dict]:
    """Four latest same-predicted-class observations per last-frame pair, never GT."""
    boxes = graph['boxes'].float()
    scale = torch.tensor([*graph['image_size_wh'], *graph['image_size_wh']], dtype=torch.float32)
    relations = torch.cat((graph['attention_logits'].softmax(-1), graph['spatial_probabilities'], graph['contact_probabilities']), -1)
    last = int(graph['pair_frames'].max())
    selected = (graph['pair_frames'] == last).nonzero().flatten().tolist()
    values, duplicate_choices = [], 0
    for pair_index in selected:
        target_box = graph['pairs'][pair_index, 1]
        category = graph['classes'][target_box]
        target_center = (boxes[target_box, 1:3] + boxes[target_box, 3:5]) / (2 * scale[:2])
        sequence = []
        for t in range(last + 1):
            candidates = [i for i in (graph['pair_frames'] == t).nonzero().flatten().tolist()
                          if graph['classes'][graph['pairs'][i, 1]] == category]
            if not candidates:
                continue
            if len(candidates) > 1:
                duplicate_choices += 1
            def distance(i: int) -> float:
                b = boxes[graph['pairs'][i, 1], 1:]
                return float((((b[:2] + b[2:]) / (2 * scale[:2])) - target_center).square().sum())
            chosen = min(candidates, key=lambda i: (distance(i), i))
            subject, obj = graph['pairs'][chosen]
            state = torch.zeros(71)
            state[:26] = relations[chosen]
            state[26 + int(category)] = 1
            state[63:67], state[67:71] = boxes[subject, 1:] / scale, boxes[obj, 1:] / scale
            sequence.append(state)
        if not sequence:
            raise ValueError('Last-frame candidate absent from own history')
        sequence = sequence[-4:]
        values.append(torch.stack([sequence[0]] * (4 - len(sequence)) + sequence))
    return torch.stack(values), {'last_pair_indices': selected, 'duplicate_class_history_choices': duplicate_choices,
                                 'candidates': len(selected), 'fallback': 'closest_center_then_lowest_pair_index'}


def observed_bank(graph: dict, future_frame_indices: list[int]) -> dict:
    """Freeze all geometry, labels, scores and candidate order for relation-only controls."""
    if not future_frame_indices or min(future_frame_indices) <= graph['nominal_cutoff']:
        raise ValueError('Future schedule must follow observation cutoff')
    last = int(graph['pair_frames'].max())
    pairs = graph['pairs'][graph['pair_frames'] == last].long()
    first, end = int(pairs.min()), int(pairs.max()) + 1
    count, n = end - first, len(future_frame_indices)
    boxes = graph['boxes'][first:end].repeat(n, 1)
    boxes[:, 0] = torch.arange(n).repeat_interleave(count)
    pair_frames = torch.arange(n, dtype=torch.int32).repeat_interleave(len(pairs))
    labels = graph['classes'][first:end].repeat(n).long()
    scores = graph['object_scores'][first:end].repeat(n).float()
    return {'boxes': boxes, 'labels': labels, 'pred_labels': labels.clone(), 'scores': scores,
            'pred_scores': scores.clone(), 'pair_idx': (pairs - first).repeat(n, 1) + count * pair_frames[:, None],
            'im_idx': pair_frames}
