"""Local-only SceneSayer official evaluator and observed-annotation persistence."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'model/SceneSayer'))
from dataloader.action_genome.ag_dataset import AG
from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator

RATIOS = (0.3, 0.5, 0.7, 0.9)


def plain(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def dataset(filtered):
    previous = Path.cwd()
    try:
        os.chdir(BASE / 'protocol/loader_cwd')
        data = AG('test', 'full', str(BASE / 'data'), True, filtered)
    finally:
        os.chdir(previous)
    name = 'filtered_small' if filtered else 'unfiltered'
    manifest = json.loads((BASE / f'data/test_{name}.json').read_text())
    assert data._video_list == [item['frame_ids'] for item in manifest]
    return data


def evaluators(data, mode):
    return {constraint: BasicSceneGraphEvaluator(mode, data.object_classes,
        data.relationship_classes, data.attention_relationships,
        data.spatial_relationships, data.contacting_relationships,
        iou_threshold=0.0, constraint=constraint) for constraint in ('with', 'no')}


def persistence(observed_frame, future_count):
    """Only the last observed graph and the future schedule length enter prediction."""
    boxes, labels, pairs, times, attention, spatial, contact = [], [], [], [], [], [], []
    for time in range(future_count):
        human = len(boxes)
        boxes.append([time] + np.asarray(observed_frame[0]['person_bbox']).reshape(-1, 4)[0].tolist())
        labels.append(1)
        for obj in observed_frame[1:]:
            pairs.append([human, len(boxes)])
            boxes.append([time] + list(obj['bbox']))
            labels.append(obj['class'])
            times.append(time)
            a = [-20.0] * 3
            for index in obj['attention_relationship']:
                a[int(index)] = 20.0
            attention.append(a)
            spatial.append([float(i in obj['spatial_relationship']) for i in range(6)])
            contact.append([float(i in obj['contacting_relationship']) for i in range(17)])
    result = {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5),
        'labels': torch.tensor(labels), 'scores': torch.ones(len(labels)),
        'pair_idx': torch.tensor(pairs, dtype=torch.long).reshape(-1, 2),
        'im_idx': torch.tensor(times, dtype=torch.long),
        'attention_distribution': torch.tensor(attention).reshape(-1, 3),
        'spatial_distribution': torch.tensor(spatial).reshape(-1, 6),
        'contacting_distribution': torch.tensor(contact).reshape(-1, 17)}
    result['pred_labels'] = result['labels']
    result['pred_scores'] = result['scores']
    return result


def metrics(evaluator):
    raw = evaluator.result_dict
    mode = evaluator.mode
    return {str(k): {'R': float(np.mean(raw[mode + '_recall'][k])),
        'mR': float(sum(np.mean(v) if v else 0 for v in raw[mode + '_mean_recall_collect'][k]) / 26)}
        for k in (10, 20, 50)}


def triples(frame):
    result = set()
    for obj in frame[1:]:
        for key, offset, reverse in [('attention_relationship', 0, False),
                                    ('spatial_relationship', 3, True),
                                    ('contacting_relationship', 9, False)]:
            for relation in obj[key]:
                result.add((obj['class'], int(relation) + offset, reverse))
    return result


def run(filtered):
    torch.set_num_threads(1)
    data = dataset(filtered)
    cohort = 'filtered_small' if filtered else 'unfiltered'
    output = BASE / 'protocol' / cohort
    output.mkdir(exist_ok=True)
    scores = {ratio: evaluators(data, 'predcls') for ratio in RATIOS}
    counts = {ratio: {'requested_cases': 0, 'official_empty_future': 0, 'scored_cases': 0,
        'future_frames': 0, 'changed_frames': 0, 'stable_frames': 0} for ratio in RATIOS}
    auxiliary = {ratio: {kind: evaluators(data, 'predcls') for kind in ('changed', 'stable')} for ratio in RATIOS}
    with (output / 'observations.jsonl').open('w') as obs, (output / 'future_gt.jsonl').open('w') as gt, \
         (output / 'cases.jsonl').open('w') as cases, (output / 'persistence.jsonl').open('w') as predictions:
        for index, frames in enumerate(data._gt_annotations):
            for ratio in RATIOS:
                cut = math.ceil(len(frames) * ratio)
                case_id = f'{data._video_list[index][0].split("/")[0]}_{ratio}'
                count = counts[ratio]
                count['requested_cases'] += 1
                record = {'case_id': case_id, 'ratio': ratio, 'video_index': index,
                    'observed_count': cut, 'future_count': len(frames) - cut,
                    'observed_frame_ids': data._video_list[index][:cut],
                    'future_frame_ids': data._video_list[index][cut:]}
                cases.write(json.dumps(record) + '\n')
                obs.write(json.dumps({'case_id': case_id, 'frames': plain(frames[:cut])}) + '\n')
                gt.write(json.dumps({'case_id': case_id, 'frames': plain(frames[cut:])}) + '\n')
                if cut == len(frames):
                    count['official_empty_future'] += 1
                    continue
                prediction = persistence(frames[cut - 1], len(frames) - cut)
                predictions.write(json.dumps({'case_id': case_id, 'prediction': plain(prediction)}) + '\n')
                # The public launcher passes the same mutable prediction WC then NC.
                for evaluator in scores[ratio].values():
                    evaluator.evaluate_scene_graph(frames[cut:], prediction)
                count['scored_cases'] += 1
                count['future_frames'] += len(frames) - cut
                observed_triples = triples(frames[cut - 1])
                for frame in frames[cut:]:
                    kind = 'stable' if triples(frame) == observed_triples else 'changed'
                    count[kind + '_frames'] += 1
                    frame_prediction = persistence(frames[cut - 1], 1)
                    for evaluator in auxiliary[ratio][kind].values():
                        evaluator.evaluate_scene_graph([frame], frame_prediction)
            if index % 100 == 0:
                print(cohort, index, 'of', len(data._video_list), flush=True)
    result = {'method': 'observed_annotation_oracle_persistence', 'input_condition':
        'Oracle-history auxiliary: observed GT boxes/classes/relationships; more privileged than GAGS; not AGS video baseline',
        'cohort': cohort, 'videos': len(data._video_list), 'frames': sum(map(len, data._video_list)),
        'protocol': 'official_replay_WC_then_NC_IoU0', 'counts': counts,
        'metrics': {r: {c: metrics(e) for c, e in es.items()} for r, es in scores.items()},
        'relation_change_auxiliary': {r: {kind: {c: metrics(e) for c, e in es.items()}
            for kind, es in groups.items()} for r, groups in auxiliary.items()}}
    (BASE / f'reports/persistence_{cohort}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'cohort': cohort, 'counts': counts, 'metrics': result['metrics']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filtered', action='store_true')
    args = parser.parse_args()
    run(args.filtered)
