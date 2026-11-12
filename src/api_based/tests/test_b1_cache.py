# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest"]
# ///
# 실행: python -m pytest api_based/tests/test_b1_cache.py
"""실제 CLI의 원본 보호와 GT 없는 연구 추론 경계를 검증한다."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bundle_paths import RESULTS


def test_raw_prediction_equals_frozen_sol_response() -> None:
    with tempfile.TemporaryDirectory(dir=RESULTS / 'b1') as temporary:
        output = Path(temporary) / 'output'
        run = subprocess.run([sys.executable,str(ROOT / 'main.py'),'b1','baseline','predict','--sample','sgcls:00607.mp4_0.5','--output',str(output)],capture_output=True,text=True)
        assert run.returncode == 0, run.stderr
        with (ROOT / 'data/b1/sol/index.final.jsonl').open() as source:
            expected = next(r for r in map(json.loads,source) if r['sample_id']=='sgcls:00607.mp4_0.5')['prediction']
        assert json.loads((output / 'prediction.json').read_text()) == expected


def test_ours_default_recomputes_frozen_macro_direct_scores() -> None:
    sample = 'sgcls:00607.mp4_0.5'
    with tempfile.TemporaryDirectory(dir=RESULTS / 'b1') as temporary:
        output = Path(temporary) / 'output'
        run = subprocess.run([sys.executable,str(ROOT / 'main.py'),'b1','ours','evaluate','--sample',sample,'--output',str(output)],capture_output=True,text=True)
        assert run.returncode == 0, run.stderr
        with (RESULTS / 'b1/frozen/results_full_1/per_case.jsonl').open() as source:
            reference = next(r for r in map(json.loads,source) if r['schedule']['sample_id']==sample)
        actual = json.loads((output / 'per_case.jsonl').read_text())
        assert actual['statistics'] == reference['methods']['macro_direct']
        assert actual['statistics'] != reference['methods']['sol']


def test_research_predict_rejects_future_gt_input() -> None:
    with tempfile.TemporaryDirectory(dir=RESULTS / 'b1') as temporary:
        forbidden = Path(temporary) / 'future_gt.pkl'
        forbidden.write_bytes(b'untrusted input must never be opened')
        run = subprocess.run([sys.executable,str(ROOT / 'main.py'),'b1','final','predict','--research','rank_residual','--input',str(forbidden),'--output',str(Path(temporary)/'output')],capture_output=True,text=True)
        assert run.returncode != 0
        assert 'PermissionError' in run.stderr and str(forbidden) in run.stderr


def test_missing_cache_never_falls_back_to_provider() -> None:
    with tempfile.TemporaryDirectory(dir=RESULTS / 'b1') as temporary:
        absent = Path(temporary) / 'missing.pkl'
        run = subprocess.run([sys.executable,str(ROOT / 'main.py'),'b1','final','predict','--research','rank_residual','--input',str(absent),'--output',str(Path(temporary)/'output')],capture_output=True,text=True)
        assert run.returncode != 0
        assert 'FileNotFoundError' in run.stderr and str(absent) in run.stderr
