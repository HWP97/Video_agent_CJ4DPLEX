# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# 실행: python runtime_guard.py baseline/b1/check_receipts.py
"""전수 해시 검사를 마친 payload와 원 receipt의 연결을 대조한다."""
import json
from pathlib import Path
from bundle_paths import REPORTS


def verify(root: Path):
    hashes = {row['destination']:row['destination_sha256'] for row in map(json.loads,(REPORTS / 'b1/copy_manifest.jsonl').open())}
    expected = {row['sample_id'] for row in map(json.loads,(root / 'data/b1/samples.jsonl').open()) if row['future_count']}
    seen = set()
    for path in (root / 'data/b1/impartail/receipts').glob('*.json'):
        receipt = json.loads(path.read_text())
        assert receipt['status'] == 'success'
        assert receipt['sample_id'] not in seen
        seen.add(receipt['sample_id'])
        assert receipt['frozen_manifest_sha256'] == hashes['data/b1/impartail/frozen_manifest.json']
        for name in ('prediction','compact'):
            assert receipt[name+'_sha256'] == hashes['data/b1/impartail/'+receipt[name+'_path']]
    assert seen == expected
    return {'receipts':len(seen),'prediction_hash_links':len(seen),'compact_hash_links':len(seen),'missing':0}


if __name__ == '__main__':
    from bundle_paths import ROOT, RESULTS
    result = verify(ROOT)
    target = RESULTS / 'b1/qa_receipts'
    target.mkdir(exist_ok=False)
    (target / 'receipt_integrity.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
