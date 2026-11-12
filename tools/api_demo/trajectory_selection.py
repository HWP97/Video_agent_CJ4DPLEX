# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
# How to run: API/local 궤적 Demo 실행기에서 import
"""절대 픽셀 오차를 먼저 제한하고 인물·시간·원영상 다양성을 함께 선택한다."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Final, Literal

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from b3_data import Case
from errors import DemoError

SOURCES: Final = ("crowds_zara01", "crowds_students003")


@dataclass(frozen=True, slots=True)
class Quality:
    mean_px: float
    final_px: float
    max_px: float
    movement_px: float
    displacement_px: float
    baseline_mean_px: float
    future_points_inside_video: bool


def quality(case: Case) -> Quality:
    scale = np.array([.0210, .0239])
    errors = np.linalg.norm((case.ours - case.primary_truth) / scale, axis=1)
    baseline = np.linalg.norm((case.baseline - case.primary_truth) / scale, axis=1)
    path = np.vstack((case.primary_observed[-1], case.primary_truth)) / scale
    pixels = np.vstack((case.ours, case.primary_truth)) * [1 / .0210, -1 / .0239] + [360, 288]
    inside = bool(np.all((pixels >= [0, 0]) & (pixels <= [719, 575])))
    return Quality(float(errors.mean()), float(errors[-1]), float(errors.max()),
                   float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                   float(np.linalg.norm(path[-1] - path[0])), float(baseline.mean()), inside)


def passes(value: Quality) -> bool:
    return (value.mean_px <= 12 and value.final_px <= 20 and value.max_px <= 25
            and value.movement_px >= 30 and value.displacement_px >= 30 and value.future_points_inside_video)


def select_source(cases: list[Case], minimum_gap_frames: int) -> list[Case]:
    """정수 최적화로 정확히10개를 동시 선택한다. 불가능하면 기준을 바꾸지 않고 실패한다."""
    eligible = sorted((case for case in cases if passes(quality(case))), key=lambda case: case.sample_id)
    if len(eligible) < 10:
        raise DemoError("절대 오차 조건을 만족하는 궤적 후보가10개보다 적습니다")
    people = sorted({case.observation.primary_id for case in eligible})
    starts = sorted({case.observation.cutoff_frame - 80 for case in eligible})
    person_index = {pid: index for index, pid in enumerate(people)}
    matrix = lil_matrix((1 + len(people) + len(starts), len(eligible)), dtype=float)
    matrix[0, :] = 1
    costs = []
    for j, case in enumerate(eligible):
        matrix[1 + person_index[case.observation.primary_id], j] = 1
        start = case.observation.cutoff_frame - 80
        for i, time in enumerate(starts):
            if start <= time < start + minimum_gap_frames:
                matrix[1 + len(people) + i, j] = 1
        value = quality(case)
        gain = np.clip(value.baseline_mean_px - value.mean_px, -25, 25)
        costs.append(value.mean_px + .1 * value.final_px + .01 * value.max_px - .05 * gain + j * 1e-8)
    low, high = np.zeros(matrix.shape[0]), np.ones(matrix.shape[0])
    low[0] = high[0] = 10
    solved = milp(np.array(costs), integrality=np.ones(len(eligible)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix.tocsr(), low, high), options={"time_limit": 60})
    if not solved.success or solved.x is None:
        raise DemoError(f"10개 동시 선택에 실패했습니다. 임계값은 유지합니다: {solved.message}")
    chosen = sorted((case for case, flag in zip(eligible, solved.x) if flag > .5), key=lambda case: case.observation.cutoff_frame)
    assert len(chosen) == len({case.observation.primary_id for case in chosen}) == 10
    assert all(b.observation.cutoff_frame - a.observation.cutoff_frame >= minimum_gap_frames for a, b in zip(chosen, chosen[1:]))
    return chosen


def select(cases: list[Case], family: Literal["api", "local"], output: Path | None = None) -> list[Case]:
    # API의 기존 평가 자료는 각 원영상의 짧은 구간에 한정된다. 시간 간격만 별도로 고정한다.
    gaps = {"crowds_students003": 80, "crowds_zara01": 110} if family == "api" else dict.fromkeys(SOURCES, 210)
    groups = {source: select_source([case for case in cases if case.dataset == source], gaps[source]) for source in SOURCES}
    chosen = [groups[source][i] for i in range(10) for source in SOURCES]
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        records = []
        for rank, case in enumerate(chosen, 1):
            value = quality(case)
            errors = np.linalg.norm(case.ours - case.primary_truth, axis=1)
            records.append({"rank": rank, "sample_id": case.sample_id, "source": case.dataset,
                "primary_id": case.observation.primary_id, "start_frame": case.observation.cutoff_frame - 80,
                "end_frame": case.observation.future_frames[-1], "mean_error_px": value.mean_px,
                "final_error_px": value.final_px, "max_error_px": value.max_px,
                "GT_path_length_px": value.movement_px, "GT_displacement_px": value.displacement_px,
                "GT_and_ours_future_points_inside_video": value.future_points_inside_video,
                "ours_ADE": float(errors.mean()), "ours_FDE": float(errors[-1]), "max_error_source_units": float(errors.max()),
                "baseline_ADE": case.baseline_score.ade, "prediction_sha256": hashlib.sha256(case.ours.tobytes()).hexdigest()})
        report = {"family": family, "task": "Trajectory Forecasting", "selection_uses_GT": True,
            "selection_is_demo_only": True, "population_metrics_unchanged": True,
            "source_videos": 2, "cases_per_source": 10, "same_person_repeated": False,
            "thresholds_native_720x576": {"mean_error_px_max": 12, "final_error_px_max": 20, "max_error_px_max": 25,
                "GT_path_length_px_min": 30, "GT_displacement_px_min": 30},
            "minimum_start_gap_frames": gaps, "clip_window_frames": 201,
            "time_windows_disjoint": family == "local", "api_overlap_reason": "existing evaluation covers only students003130.0–164.8s and zara01215.2–276.0s" if family == "api" else None,
            "objective": "mean_px + 0.1*final_px + 0.01*max_px - 0.05*clip(baseline_mean_px-mean_px,-25,25), plus deterministic tiny tie break",
            "counts": {source: {"candidates": sum(case.dataset == source for case in cases),
                                  "eligible": sum(case.dataset == source and passes(quality(case)) for case in cases)} for source in SOURCES},
            "records": records}
        (output / "selection_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return chosen
