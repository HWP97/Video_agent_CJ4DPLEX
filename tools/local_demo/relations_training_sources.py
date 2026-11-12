# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# 실행: python tools/local_demo/relations_training_sources.py
"""선택 모델의 학습·입력·선택 소스를 원문 보존 복사한다."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent/'HWP_CJ4DPLEX_AI_support/iterative_research_20261111'
DEST = ROOT/'src/local_model_based/data/b1/latest'
FILES = ('b1/models.py','b1/prepare.py','b2/models.py',
    'b1/combinations_81112/train.py','b1/combinations_81112/models.py',
    'b1/combinations_81112/prepare.py','b1/combinations_81112/evaluate.py',
    'b1/combinations_81112/lock_selection.py','b1/combinations_81112/data/receipt.json',
    'b1/combinations_81112/selection_lock.json','b1/combinations_81112/combined_selection_lock.json')


def main() -> None:
    records = []
    for name in FILES:
        src, dst = SOURCE/name, DEST/'source'/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        before = hashlib.sha256(src.read_bytes()).hexdigest()
        assert hashlib.sha256(dst.read_bytes()).hexdigest()==before
        records.append({'source':str(src),'destination':str(dst.relative_to(DEST)),'sha256':before})
    (DEST/'training_sources.json').write_text(json.dumps(records,indent=2))
    print(json.dumps({'copied':len(records),'training_runs':0}))


if __name__=='__main__':
    main()
