# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pydantic", "opencv-python-headless"]
# ///
# How to run: 기존 환경 Python -B tools/local_demo/trajectory_data.py
"""최신 동결 로컬 예측에서 실제 영상으로 확인할20사례를 선택한다."""
from pathlib import Path
import json
import sys
from typing import Final
import numpy as np
from scipy.interpolate import interp1d

ROOT: Final = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/api_demo"))
from b3_data import Case, Observation, Track, Truth, Score
from b3_projection import load_sources, project
from b3_source_check import splines

DATA: Final = ROOT / "src/local_model_based/data/b3/latest"
OUT: Final = ROOT / "demo/local_model_based/trajectory_forecasting"


def load_cases() -> list[Case]:
    sources = {source.dataset: source for source in load_sources()}
    manifest = json.loads((DATA / "manifest.json").read_text())
    candidates = []
    for fold in manifest["folds"]:
        baseline = np.load(DATA / f"reference/baseline_fold{fold['fold']}.npy", mmap_mode="r")
        ours = np.load(DATA / f"reference/ours_fold{fold['fold']}.npy", mmap_mode="r")
        offset = 0
        for stem in fold["stems"]:
            rows = json.loads((DATA / f"indices/{stem}_index.json").read_text())
            if stem not in sources:
                offset += len(rows)
                continue
            with np.load(DATA / f"ground_truth/raw/{stem}.npz") as raw:
                primary_ids = {int(row[0]): int(row[1]) for row in raw["scenes"]}
                tracks = raw["tracks"].copy()
            with np.load(DATA / f"ground_truth/{stem}_labels.npz") as labels:
                truth = labels["xy"].copy()
            with np.load(DATA / f"ground_truth/scores/{stem}.npz") as saved:
                methods = saved["methods"].tolist()
                b, o = methods.index("baseline_attention"), methods.index("B3-D04")
                ade, fde, collision = saved["ADE"].copy(), saved["FDE"].copy(), saved["collision"].copy()
            for i, row in enumerate(rows):
                first, cutoff, end = row["observed_frames"][0], row["observed_frames"][-1], row["future_frames"][-1]
                if cutoff - first != 80 or end - cutoff != 120:
                    continue
                pid = primary_ids[int(row["sample_id"].rsplit("_", 1)[1])]
                path = tracks[(tracks[:, 1] == pid) & np.isin(tracks[:, 0], row["observed_frames"])]
                path = path[np.argsort(path[:, 0])]
                if len(path) != 9 or not np.array_equal(path[:, 0], row["observed_frames"]):
                    continue
                observation = Observation(sample_id=row["sample_id"], primary_id=pid, dt_s=.4, cutoff_s=3.2,
                    cutoff_frame=cutoff, tracks=[Track(pedestrian_id=pid, points=[(int(p[0]), (p[0] - first) / 25, p[2], p[3]) for p in path])],
                    future_frames=row["future_frames"], future_times_s=[3.2 + .4 * k for k in range(1, 13)])
                target = Truth(sample_id=row["sample_id"], paths=[[(frame, pid, float(x), float(y)) for frame, (x, y) in zip(row["future_frames"], truth[i])]])
                score0 = Score(sample_id=row["sample_id"], ade=float(ade[b, i]), fde=float(fde[b, i]), collision_gt=int(collision[b, i, 1]))
                score1 = Score(sample_id=row["sample_id"], ade=float(ade[o, i]), fde=float(fde[o, i]), collision_gt=int(collision[o, i, 1]))
                candidates.append(Case(row["sample_id"], stem, observation, target, baseline[offset+i], ours[offset+i], score0, score1))
            offset += len(rows)
    return candidates


def select() -> list[Case]:
    from trajectory_selection import select as select_quality
    return select_quality(load_cases(), "local", OUT)


def prepare() -> list[Case]:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = select()
    sources = {source.dataset: source for source in load_sources()}
    raw = ROOT.parent / "my_works/experiments/benchmarks/2026-11-06/001_frozen_full/b3/data/author_conversion/data/raw/crowds"
    curves = {name: splines(raw / ("students003.vsp" if name == "crowds_students003" else name + ".vsp")) for name in sources}
    checks, rows = [], []
    for case in cases:
        observed = case.observation.tracks[0].points
        frames = [p[0] for p in observed] + case.observation.future_frames
        points = np.vstack((case.primary_observed, case.primary_truth))
        curve = curves[case.dataset][case.observation.primary_id]
        kind = "cubic" if len(curve) > 5 else "linear"
        expected = np.column_stack([interp1d(curve[:, 2], curve[:, axis], kind=kind)(frames) for axis in range(2)]) * [.0210, .0239]
        error = float(np.max(np.abs(points - expected)))
        assert error <= .005000001, (case.sample_id, error)
        checks.append({"sample_id": case.sample_id, "primary": case.observation.primary_id, "points": 21, "max_axis_error": error})
        rows.append({"sample_id": case.sample_id, "dataset": case.dataset, "primary_id": case.observation.primary_id,
                     "observed_frames": [p[0] for p in observed], "future_frames": case.observation.future_frames,
                     "observed": case.primary_observed.tolist(), "baseline": case.baseline.tolist(), "ours": case.ours.tolist(),
                     "truth": case.primary_truth.tolist(), "baseline_ADE": case.baseline_score.ade, "ours_ADE": case.ours_score.ade})
    (OUT / "cases.json").write_text(json.dumps(rows, indent=2))
    (OUT / "projection_verification.json").write_text(json.dumps({"verified": True, "points": 420, "frame_offset": 0,
        "projection": "u=x/0.0210+360, v=288-y/0.0239", "records": checks}, indent=2))
    case = cases[0]
    source = sources[case.dataset]
    payload = {"sample_id": case.sample_id, "task": "Trajectory Forecasting", "method": "B3-D04",
        "source_video": str(source.path), "source_fps": 25, "width": 720, "height": 576,
        "primary_id": case.observation.primary_id, "start_frame": case.observation.cutoff_frame - 80,
        "cutoff_frame": case.observation.cutoff_frame, "end_frame": case.observation.future_frames[-1],
        "observed_seconds": 3.2, "future_seconds": 4.8, "observed_world": case.primary_observed.tolist(),
        "observed_pixels": project(case.primary_observed, (720, 576)).tolist(),
        "baseline_world": case.baseline.tolist(), "ours_world": case.ours.tolist(),
        "baseline_pixels": project(case.baseline, (720, 576)).tolist(), "ours_pixels": project(case.ours, (720, 576)).tolist(),
        "GT_in_prompt": False}
    (OUT / "cosmos_input.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"cases": len(cases), "representative": case.sample_id, "maximum_projection_error": max(row["max_axis_error"] for row in checks)}))
    return cases


if __name__ == "__main__":
    prepare()
