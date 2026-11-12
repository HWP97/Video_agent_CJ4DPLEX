# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# 실행: python tools/local_demo/relations_prepare.py
"""최신 관계 head 자산을 복사하고 원본 영상별 Demo 선택을 고정한다."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / 'HWP_CJ4DPLEX_AI_support/iterative_research_20261111/b1/combinations_81112'
DEST = ROOT / 'src/local_model_based/data/b1/latest'
SELECTED = {'sgcls': 'c04_gate_positive_cap3', 'predcls': 'c06_gate_positive_cap15'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    sources = ['combined_selection_lock.json', 'RESULTS.md', 'evaluation/test/summary.json']
    for mode, method in SELECTED.items():
        sources += [f'full/{mode}_{method}.pt', f'full/{mode}_{method}.json', f'data/{mode}_memory.npz']
    records = []
    for relative in sources:
        src = SOURCE / relative
        dst = DEST / src.name
        shutil.copy2(src, dst)
        assert digest(src) == digest(dst)
        records.append({'source': str(src), 'destination': dst.name, 'sha256': digest(src)})
    candidates = []
    videos = ROOT.parent / 'my_works/b1/benchmarks/action_genome_20261106/data/videos'
    for path in sorted((SOURCE / 'evaluation/test').glob('*__*.json')):
        row = json.loads(path.read_text())
        if row.get('ratio') != .5 or not (videos / row['video']).is_file():
            continue
        base = row['methods']['baseline']
        ours = row['methods'][SELECTED[row['mode']]]
        score = lambda values, key: values[key]['R_sum'][1] / values[key]['frames']
        candidates.append({'sample_id': row['sample_id'], 'mode': row['mode'], 'video_id': row['video'],
            'baseline': score(base, 'no'), 'ours': score(ours, 'no'),
            'baseline_wc': score(base, 'with'), 'ours_wc': score(ours, 'with')})
    chosen, seen = [], set()
    for row in sorted(candidates, key=lambda r: (r['baseline'] - r['ours'], r['sample_id'])):
        if row['video_id'] in seen:
            continue
        chosen.append(row)
        seen.add(row['video_id'])
        if len(chosen) == 20:
            break
    assert len(chosen) == 20
    for row in chosen:
        stem = row['sample_id'].replace(':', '__')
        for extension in ('.pt', '.json'):
            src = SOURCE / 'evaluation/test' / (stem + extension)
            dst = DEST / 'reference' / src.name
            dst.parent.mkdir(exist_ok=True)
            shutil.copy2(src, dst)
            records.append({'source': str(src), 'destination': str(dst.relative_to(DEST)), 'sha256': digest(src)})
    (DEST / 'demo_selection.json').write_text(json.dumps(chosen, indent=2))
    (DEST / 'provenance.json').write_text(json.dumps(records, indent=2))
    print(json.dumps({'assets': len(records), 'candidates': len(candidates), 'demos': len(chosen)}))


if __name__ == '__main__':
    main()
