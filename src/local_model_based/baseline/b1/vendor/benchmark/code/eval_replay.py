"""Cache-only full-manifest replay; missing or invalid outputs remain zero-recall cases."""
import argparse
import json
from collections import Counter
from itertools import zip_longest
from types import SimpleNamespace

import numpy as np
import torch
from eval_ag import BASE, RATIOS, evaluators, metrics, persistence
from sol_ag_adapter import parse_prediction


def restore(frames):
    for frame in frames:
        frame[0]['person_bbox'] = np.asarray(frame[0]['person_bbox'])
        for obj in frame[1:]:
            obj['bbox'] = np.asarray(obj['bbox'])
            for key in ('attention_relationship', 'spatial_relationship', 'contacting_relationship'):
                obj[key] = torch.as_tensor(obj[key], dtype=torch.long)
    return frames


def replay(cohort, mode, prediction_path, output):
    torch.set_num_threads(1)
    protocol = BASE / 'protocol' / cohort
    # Public evaluator consumes numerical classes; names only map group-local relation IDs.
    relations = [str(i) for i in range(26)]
    vocab = SimpleNamespace(object_classes=[str(i) for i in range(37)],
        relationship_classes=relations, attention_relationships=relations[:3],
        spatial_relationships=relations[3:9], contacting_relationships=relations[9:])
    scores = {ratio: evaluators(vocab, mode) for ratio in RATIOS}
    counts = {ratio: Counter() for ratio in RATIOS}
    offsets = {}
    with prediction_path.open() as predictions:
        while True:
            offset = predictions.tell()
            line = predictions.readline()
            if not line:
                break
            case_id = json.loads(line)['case_id']
            if case_id in offsets:
                raise ValueError('duplicate_case:' + case_id)
            offsets[case_id] = offset
        seen = set()
        with (protocol / 'cases.jsonl').open() as cases, (protocol / 'future_gt.jsonl').open() as truths:
            for case_line, gt_line in zip_longest(cases, truths):
                if case_line is None or gt_line is None:
                    raise ValueError('manifest_gt_length_mismatch')
                case, truth = json.loads(case_line), json.loads(gt_line)
                assert case['case_id'] == truth['case_id']
                assert len(truth['frames']) == case['future_count']
                seen.add(case['case_id'])
                ratio = case['ratio']
                counts[ratio]['requested'] += 1
                if not case['future_count']:
                    counts[ratio]['official_empty_future'] += 1
                    continue
                future = restore(truth['frames'])
                assert len(future) == case['future_count']
                try:
                    if case['case_id'] not in offsets:
                        raise ValueError('missing_prediction')
                    predictions.seek(offsets[case['case_id']])
                    cached = json.loads(predictions.readline())
                    pred = parse_prediction(cached['prediction'], case['future_count'], mode)
                    counts[ratio]['valid'] += 1
                except (ValueError, KeyError, TypeError, RuntimeError) as error:
                    counts[ratio]['failure:' + str(error)] += 1
                    pred = persistence([], 0)
                for evaluator in scores[ratio].values():
                    evaluator.evaluate_scene_graph(future, pred)
                counts[ratio]['evaluated_future_frames'] += len(future)
    expected = json.loads((BASE / 'protocol/denominators.json').read_text())[cohort]
    for ratio in RATIOS:
        assert counts[ratio]['requested'] == expected[str(ratio)]['requested_cases']
        assert counts[ratio]['evaluated_future_frames'] == expected[str(ratio)]['future_frames']
    if set(offsets) - seen:
        raise ValueError('unknown_prediction_cases')
    result = {'cohort': cohort, 'mode': mode, 'counts': counts, 'metrics':
        {ratio: {key: metrics(value) for key, value in group.items()} for ratio, group in scores.items()},
        'protocol': 'official_replay_WC_then_NC_IoU0', 'paid_calls': 0}
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'output': str(output), 'counts': counts, 'paid_calls': 0}), flush=True)


if __name__ == '__main__':
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', choices=['filtered_small', 'unfiltered'], required=True)
    parser.add_argument('--mode', choices=['sgdet', 'sgcls', 'predcls'], required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    replay(args.cohort, args.mode, args.predictions, args.output)
