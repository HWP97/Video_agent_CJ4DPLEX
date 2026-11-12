# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pydantic", "opencv-python-headless"]
# ///
# 실행: 기존 b1 환경 Python으로 tools/api_demo/b2.py 실행
"""동결 b2 예측에서 같은 final 사례의 데모 후보를 고른다."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict
from errors import DemoError

ROOT: Final = Path(__file__).resolve().parents[2]
API: Final = ROOT / "src/api_based"
DATA: Final = API / "data/b2"
OUT: Final = ROOT / "demo/api_based/affect_forecasting"
VIDEO_ROOT: Final = Path(os.environ.get("VEATIC_VIDEO_ROOT", str(
    ROOT.parent / "my_works/b2/results/2026-11-05/002_veatic_forecast/data/VEATIC/videos")))
HORIZONS: Final = (0.5, 1.0, 2.0, 3.0)


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True)
    sample_id: str
    group: str
    split: str
    cutoff_s: float
    baseline: list[list[float]]
    current: list[float]
    local: list[float]


class Truth(BaseModel):
    model_config = ConfigDict(frozen=True)
    sample_id: str
    horizon_s: float
    pts_s: float
    VA: tuple[float, float]


class Case(BaseModel):
    model_config = ConfigDict(frozen=True)
    sample_id: str
    video_id: int
    group: str
    rank: int
    cutoff_s: float
    baseline: list[list[float]]
    ours: list[list[float]]
    truth: list[list[float]]
    gt_pts: list[float]
    baseline_metric: float
    ours_metric: float
    metric_delta: float
    target_description: str


def select_cases() -> list[Case]:
    """학습/선정 사례를 제외하고 원본 영상별 최대 개선 구간을 고른다."""
    sys.path.insert(0, str(API / "ours/b2"))
    from ridge_model import Inputs, predict

    rows = [Observation.model_validate(r) for r in json.loads((DATA / "ridge/inputs.json").read_text())]
    frozen = json.loads((DATA / "ridge/frozen_model.json").read_text())
    model = frozen["models"]["observation_residual_lambda1"]
    inputs = Inputs(np.array([r.baseline for r in rows]), np.array([r.current for r in rows]),
                    np.array([r.local for r in rows]), np.array(HORIZONS))
    _, corrected = predict(inputs, model)
    # GT는 보정 예측을 확정한 다음 선정·평가에서만 읽는다.
    truth = [Truth.model_validate(r) for r in json.loads((DATA / "ground_truth/forecast.json").read_text())]
    lookup = {(r.sample_id, r.horizon_s): r for r in truth}
    descriptions = {r["video_id"]: r["target_description"] for r in json.loads(
        (DATA / "inputs/identity_review/descriptions.json").read_text())}
    candidates = []
    for row, ours in zip(rows, corrected):
        if row.split != "final":
            continue
        gt = [lookup[row.sample_id, h] for h in HORIZONS]
        y = np.array([r.VA for r in gt])
        baseline_mae = float(np.abs(np.array(row.baseline) - y).mean())
        ours_mae = float(np.abs(ours - y).mean())
        video_id = int(row.sample_id[1:4])
        candidates.append(Case(sample_id=row.sample_id, video_id=video_id, group=row.group,
            rank=0, cutoff_s=row.cutoff_s, baseline=row.baseline,
            ours=ours.tolist(), truth=y.tolist(), gt_pts=[r.pts_s for r in gt],
            baseline_metric=baseline_mae, ours_metric=ours_mae,
            metric_delta=baseline_mae - ours_mae, target_description=descriptions[video_id]))
    chosen: list[Case] = []
    used: set[str] = set()
    for case in sorted(candidates, key=lambda c: (-c.metric_delta, c.sample_id)):
        if case.group in used or case.metric_delta <= 0:
            continue
        chosen.append(case.model_copy(update={"rank": len(chosen)+1}))
        used.add(case.group)
        if len(chosen) == 20:
            break
    if len(chosen) != 20:
        raise DemoError(f"demo 수 부족: {len(chosen)}/20")
    return chosen
