# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# How to run: 기존 b1 model/env/bin/python -B prepare.py
"""관측 metadata만으로 실험과 분할을 등록한다."""
from pathlib import Path
import hashlib
import json
import math
import sys
from typing import Final
import numpy as np

OUT: Final = Path(__file__).resolve().parent
ROOT: Final = OUT.parents[1]
SOURCE: Final = ROOT.parent / 'official_local_train_20261111/b2/attempt_fp32'
sys.path.insert(0, str(ROOT))
from followup_common import sha, save, guard


def main() -> None:
    guard(OUT, True)
    assert not (OUT / 'registration.json').exists()
    rows = json.loads((SOURCE / 'manifest.json').read_text())
    old = json.loads((ROOT / 'b2/data/receipt.json').read_text())
    train = [i for i, r in enumerate(rows) if r['split'] == 'train']
    validation_groups = set(old['inner_validation_groups'])
    group_val = [i for i in train if rows[i]['group'] in validation_groups]
    inner = [i for i in train if i not in group_val]
    fit, temporal, purge = [], [], []
    for group in sorted({rows[i]['group'] for i in inner}):
        ids = sorted([i for i in inner if rows[i]['group'] == group], key=lambda i: rows[i]['cutoff_s'])
        n = max(1, math.ceil(len(ids) * .2))
        held = ids[-n:]
        temporal.extend(held)
        boundary = min(rows[i]['observed_start_s'] for i in held)
        for i in ids[:-n]:
            (fit if rows[i]['cutoff_s'] + max(rows[i]['horizons_s']) < boundary else purge).append(i)
    assert len(group_val) == 256
    assert not set(fit) & (set(group_val) | set(temporal) | set(purge))
    for a in fit:
        for b in temporal:
            if rows[a]['group'] == rows[b]['group']:
                assert rows[a]['cutoff_s'] + max(rows[a]['horizons_s']) < rows[b]['observed_start_s']
    samples = {}
    for name, ids in [('fit', fit), ('inner_group', group_val), ('official_test', [i for i,r in enumerate(rows) if r['split'] == 'official_test'])]:
        samples[name] = sorted(ids, key=lambda i: hashlib.sha256(('20261111:' + rows[i]['sample_id']).encode()).hexdigest())[:3]
    save(OUT / 'split.json', {'fit': sorted(fit), 'inner_group': sorted(group_val), 'inner_time': sorted(temporal), 'purged': sorted(purge),
        'train_full': train, 'sample_ids': {k: [rows[i]['sample_id'] for i in v] for k,v in samples.items()},
        'forward_samples': samples, 'labels_read': False, 'purge_window': 'observed_start_s through cutoff_s+max(horizons_s)'})
    files = [SOURCE / n for n in ('manifest.json','inputs.npz','assembly_receipt.json','extraction_complete.json','extract.py','labels_train.json')]
    files += [ROOT / 'b2/data/receipt.json', ROOT / 'b2/followup_reasoned_combinations/selection_lock.json', ROOT / 'b2/followup_reasoned_combinations/COMPLETION_RECEIPT.json']
    files += [SOURCE / 'predictions/direct_gru_h8_contrast0.npz']
    save(OUT / 'registration.json', {'seed': 20261111, 'configs': ['B2-D01','B2-D02','B2-D03'], 'conditional': 'B2-D04',
        'ridge': .01, 'pca_rank': 16, 'epochs': 'analytic one pass', 'fit_counts': {k:len(v) for k,v in [('fit',fit),('inner_group',group_val),('inner_time',temporal),('purged',purge)]},
        'protected': {str(p):sha(p) for p in files}, 'decisions_sha256':sha(OUT / 'DECISIONS.md'), 'split_sha256':sha(OUT/'split.json'), 'source':sha(Path(__file__)),
        'selection': 'both validation future MAE<=hold and CCC>=hold; minimize worst relative MAE; else hold', 'outer_labels_read':False})
    print(json.dumps({'counts':json.loads((OUT/'registration.json').read_text())['fit_counts'], 'samples':samples}), flush=True)


if __name__ == '__main__':
    main()
