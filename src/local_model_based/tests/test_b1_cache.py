# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest", "numpy", "torch"]
# ///
# 실행: python -m pytest local_model_based/tests/test_b1_cache.py
"""고정 baseline 보존과 최신 선택 head의 독립 원실험 출력 재현을 검증한다."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bundle_paths import RESULTS


SOURCE = ROOT.parents[2] / 'HWP_CJ4DPLEX_AI_support/iterative_research_20261111/b1/combinations_81112/evaluation/test'
SAMPLE = 'predcls:GF7DV.mp4_0.5'
STEM = SAMPLE.replace(':', '__')


def test_baseline_evaluation_preserves_original_impartail_scores() -> None:
    expected = json.loads((SOURCE / f'{STEM}.json').read_text())['methods']['baseline']
    with tempfile.TemporaryDirectory(dir=RESULTS / 'b1') as temporary:
        output = Path(temporary) / 'baseline'
        run = subprocess.run([sys.executable,str(ROOT / 'main.py'),'b1','baseline','evaluate','--sample',SAMPLE,'--output',str(output)],capture_output=True,text=True)
        assert run.returncode == 0, run.stderr
        actual = json.loads((output/'per_case.jsonl').read_text())['statistics']
        assert actual == expected


def test_final_cli_recomputes_latest_selected_head_probabilities() -> None:
    original = torch.load(SOURCE / f'{STEM}.pt', map_location='cpu', weights_only=False)
    expected = original['probabilities']['c06_gate_positive_cap15'].numpy()
    with tempfile.TemporaryDirectory(dir=RESULTS / 'b1') as temporary:
        output = Path(temporary) / 'latest'
        run = subprocess.run([sys.executable,str(ROOT / 'main.py'),'b1','final','predict','--sample',SAMPLE,'--output',str(output)],capture_output=True,text=True)
        assert run.returncode == 0, run.stderr
        actual = torch.load(output / f'{STEM}.pt', map_location='cpu', weights_only=False)
        np.testing.assert_allclose(actual['probabilities'].numpy(), expected, rtol=0, atol=1e-7)


def test_unknown_sample_fails_instead_of_empty_success() -> None:
    with tempfile.TemporaryDirectory(dir=RESULTS / 'b1') as temporary:
        run = subprocess.run([sys.executable,str(ROOT / 'main.py'),'b1','final','evaluate','--sample','predcls:missing.mp4_0.5','--output',str(Path(temporary)/'output')],capture_output=True,text=True)
        assert run.returncode != 0
        assert 'KeyError' in run.stderr and 'predcls:missing.mp4_0.5' in run.stderr
        assert not (Path(temporary)/'output/receipt.json').exists()
