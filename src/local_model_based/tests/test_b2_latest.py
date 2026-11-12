# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pydantic", "pytest"]
# ///
# 실행: 기존 model/env/bin/python -B -m pytest tests/test_b2_latest.py
"""동결 원실험 출력과 새 관측 전용 D04 실행의 수치 계약."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ours/b2"))
import latest_inference


@pytest.mark.parametrize("method", ["B2-C01", "B2-D04"])
def test_frozen_model_recomputed_without_target_access_matches_original(method: latest_inference.Method) -> None:
    with np.load(latest_inference.DATA / f"reference/{method}_prediction.npz") as saved:
        expected_current = saved["current"] if "current" in saved.files else None
        expected_prediction = saved["prediction"]
    current, prediction, blocked = latest_inference.predict(method)
    assert blocked
    if expected_current is None:
        assert current is None
    else:
        np.testing.assert_allclose(current, expected_current, rtol=0, atol=1e-12)
    np.testing.assert_allclose(prediction, expected_prediction, rtol=0, atol=1e-12)
