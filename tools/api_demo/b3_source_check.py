# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pydantic", "opencv-python-headless"]
# ///
# 실행: 기존 환경 Python으로 tools/api_demo/b3_source_check.py
"""선정된 보행자의 원 VSP 제어점과 현재 관측·정답 좌표를 대조한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from b3_data import Array, ROOT, load_cases, select_cases
from b3_projection import OUTPUT, load_sources


def splines(path: Path) -> list[Array]:
    curves = []
    current = []
    for line in path.read_text().splitlines():
        if "Num of control points" in line or "the number of splines" in line:
            if current:
                curves.append(np.asarray(current))
                current = []
            continue
        values = line.split(" - ", 1)[0].split()
        if len(values) == 4:
            current.append([float(values[0]), float(values[1]), int(values[2])])
    if current:
        curves.append(np.asarray(current))
    return curves


def main() -> None:
    sources = load_sources()
    selected = select_cases(load_cases(), frozenset(source.dataset for source in sources))
    raw = ROOT.parent / "my_works/experiments/benchmarks/2026-11-06/001_frozen_full/b3/data/author_conversion/data/raw/crowds"
    names = {"crowds_students003": "students003.vsp", "crowds_zara01": "crowds_zara01.vsp"}
    original = {name: splines(raw / filename) for name, filename in names.items()}
    evidence = []
    for case in selected:
        control = original[case.dataset][case.observation.primary_id]
        observed = next(t for t in case.observation.tracks if t.pedestrian_id == case.observation.primary_id)
        frames = [p[0] for p in observed.points] + case.observation.future_frames
        current = np.vstack((case.primary_observed, case.primary_truth))
        kind = "cubic" if len(control) > 5 else "linear"
        native = np.column_stack([interp1d(control[:, 2], control[:, axis], kind=kind)(frames) for axis in range(2)])
        source_world = native * np.array([.0210, .0239])
        error = float(np.max(np.abs(current-source_world)))
        assert error <= .005000001
        evidence.append({"sample_id": case.sample_id, "source": str(raw / names[case.dataset]),
                         "pedestrian_id": case.observation.primary_id, "points_checked": len(frames),
                         "max_axis_error_m": error, "frame_offset": 0})
    path = OUTPUT / ".omo/evidence/original_annotation_check.json"
    path.write_text(json.dumps({"verified": True, "cases": len(evidence), "points": sum(r["points_checked"] for r in evidence),
                               "rounding_tolerance_m": .005, "records": evidence}, indent=2), encoding="utf-8")
    print(json.dumps({"verified": True, "cases": len(evidence), "points": 21*len(evidence),
                      "maximum_axis_error_m": max(r["max_axis_error_m"] for r in evidence)}))


if __name__ == "__main__":
    main()
