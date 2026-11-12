# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest", "numpy"]
# ///
# How to run: existing environment python -m pytest tests/test_b3_contract.py
"""실제 CLI의 누락 캐시 실패와 고정 baseline 계약."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bundle_paths import RESULTS

def test_missing_response_fails_without_paid_fallback() -> None:
    response = next((ROOT / "data/b3/sol_cache").glob("*/response.json"))
    hidden = response.with_suffix(".hidden")
    response.rename(hidden)
    try:
        with tempfile.TemporaryDirectory(dir=RESULTS / "b3") as folder:
            run = subprocess.run([sys.executable, str(ROOT / "main.py"), "b3", "baseline", "replay",
                "--output", str(Path(folder) / "run")], capture_output=True, text=True, check=False)
        assert run.returncode != 0
        assert "FileNotFoundError" in run.stderr and str(response) in run.stderr
    finally:
        hidden.rename(response)

def test_baseline_rejects_research_method() -> None:
    run = subprocess.run([sys.executable, str(ROOT / "main.py"), "b3", "baseline", "predict",
        "--research-option"], capture_output=True, text=True, check=False)
    assert run.returncode == 2
    assert "baseline은 raw Sol 고정" in run.stderr

def test_ours_default_is_fixed_blend_and_preserves_missing() -> None:
    with tempfile.TemporaryDirectory(dir=RESULTS / "b3") as folder:
        output = Path(folder) / "ours"
        run = subprocess.run([sys.executable, str(ROOT / "main.py"), "b3", "ours", "predict",
            "--output", str(output)], capture_output=True, text=True, check=False)
        assert run.returncode == 0, run.stderr
        with np.load(output / "predictions.npz") as values:
            actual = values["positions"]
    with np.load(ROOT / "data/b3/inputs.npz") as values:
        expected = .75 * values["sol"] + .25 * values["cv"]
    np.testing.assert_array_equal(actual, expected)
    assert int(np.isfinite(actual).all((1, 2)).sum()) == 1939

