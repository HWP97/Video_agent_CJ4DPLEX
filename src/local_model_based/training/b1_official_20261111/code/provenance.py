# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# 실행: 학습/평가 모듈이 호출
"""재개 시 동결 입력 변경과 미확인 checkpoint 재사용을 거부한다."""
from pathlib import Path
import hashlib
import json


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(path: Path, value) -> str:
    if path.exists():
        assert json.loads(path.read_text())==json.loads(json.dumps(value)), f'Frozen provenance mismatch: {path}'
    else:
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
    return digest(path)


def training_signature(here: Path,source: Path,stage: str):
    inputs=[source/'protocol/split.json',source/'protocol/train_manifest.json',*sorted((source/'assets/compact_train').glob('*.pt')),*sorted((source/'assets/supervision').glob('*.json'))]
    return {'stage':stage,'runplan':digest(here/'runplan.json'),'code':{p.name:digest(p) for p in (here/'code').glob('*.py')},
            'inputs':{str(p):digest(p) for p in inputs}}
