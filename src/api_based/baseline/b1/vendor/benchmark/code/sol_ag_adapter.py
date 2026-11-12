"""Offline-only cache parser. No provider client or transmission is permitted here."""
import torch


def parse_prediction(value, future_count, mode='sgdet'):
    """Validate cached native evaluator arrays without consulting future annotations."""
    sizes = {'boxes': 5, 'pair_idx': 2, 'attention_distribution': 3,
             'spatial_distribution': 6, 'contacting_distribution': 17}
    if mode == 'predcls':
        value = dict(value, pred_labels=value['labels'], pred_scores=value['scores'])
    result = {}
    for key in (*sizes, 'im_idx', 'pred_labels', 'pred_scores'):
        dtype = torch.long if key in ('pair_idx', 'im_idx', 'pred_labels') else torch.float32
        raw = value[key]
        if key in sizes and any(not isinstance(row, list) or len(row) != sizes[key] for row in raw):
            raise ValueError(f'shape:{key}')
        if dtype == torch.long and any(float(x) != int(x) for row in raw for x in
                                      (row if isinstance(row, list) else [row])):
            raise ValueError(f'noninteger:{key}')
        tensor = torch.tensor(raw, dtype=dtype)
        if key in sizes:
            tensor = tensor.reshape(-1, sizes[key])
        elif tensor.ndim != 1:
            raise ValueError(f'shape:{key}')
        if not torch.isfinite(tensor).all():
            raise ValueError(f'nonfinite:{key}')
        result[key] = tensor
    objects = len(result['boxes'])
    pairs = len(result['pair_idx'])
    if objects != len(result['pred_labels']) or objects != len(result['pred_scores']):
        raise ValueError('object_length')
    if any(len(result[key]) != pairs for key in ('im_idx', 'attention_distribution',
            'spatial_distribution', 'contacting_distribution')):
        raise ValueError('pair_length')
    if pairs and (result['pair_idx'].min() < 0 or result['pair_idx'].max() >= objects):
        raise ValueError('pair_index')
    if pairs and (result['im_idx'].min() < 0 or result['im_idx'].max() >= future_count):
        raise ValueError('future_index')
    if objects and ((result['pred_labels'] < 1).any() or (result['pred_labels'] > 36).any()):
        raise ValueError('class_range')
    for key in ('pred_scores', 'spatial_distribution', 'contacting_distribution'):
        if ((result[key] < 0) | (result[key] > 1)).any():
            raise ValueError(f'probability_range:{key}')
    boxes = result['boxes']
    if objects and (boxes[:, 3:] < boxes[:, 1:3]).any():
        raise ValueError('bbox_range')
    # Official ODE sets the entire boxes tensor, including column0, to0.5.
    # Its evaluator ignores column0; im_idx alone identifies future frames.
    result['labels'] = result['pred_labels']
    result['scores'] = result['pred_scores']
    return result
