"""현재 API 궤적 결과를 읽고 같은 사례의 데모20개를 선정한다."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

ROOT: Final = Path(__file__).resolve().parents[2]
DATA: Final = ROOT / "src/api_based/data/b3"
RESULTS: Final = ROOT / "results/api_based/b3"
Array = NDArray[np.float64]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Score(Frozen):
    sample_id: str
    ade: float | None
    fde: float | None
    collision_gt: int | None


class Track(Frozen):
    pedestrian_id: int
    points: list[tuple[int, float, float, float]]


class Observation(Frozen):
    sample_id: str
    primary_id: int
    dt_s: float
    cutoff_s: float
    cutoff_frame: int
    tracks: list[Track]
    future_frames: list[int]
    future_times_s: list[float]


class Truth(Frozen):
    sample_id: str
    paths: list[list[tuple[int, int, float, float]]]


class Index(Frozen):
    sample_id: str
    dataset: str
    primary_id: int


@dataclass(frozen=True, slots=True)
class Case:
    sample_id: str
    dataset: str
    observation: Observation
    truth: Truth
    baseline: Array
    ours: Array
    baseline_score: Score
    ours_score: Score

    @property
    def improvement(self) -> float:
        assert self.baseline_score.ade is not None and self.ours_score.ade is not None
        return self.baseline_score.ade - self.ours_score.ade

    @property
    def primary_observed(self) -> Array:
        track = next(t for t in self.observation.tracks if t.pedestrian_id == self.observation.primary_id)
        return np.asarray(track.points, dtype=float)[:, 2:]

    @property
    def primary_truth(self) -> Array:
        return np.asarray(self.truth.paths[0], dtype=float)[:, 2:]


def load_cases() -> list[Case]:
    """원 예측 배열과 저장 점수를 대조하고 유효한 같은 집합만 반환한다."""
    indices = [Index.model_validate(r) for r in json.loads((DATA / "index.json").read_text())]
    observations = {r.sample_id: r for r in (
        Observation.model_validate_json(line) for line in (DATA / "observations.jsonl").read_text().splitlines())}
    truths = {r.sample_id: r for r in (
        Truth.model_validate_json(line) for line in (DATA / "ground_truth/futures.jsonl").read_text().splitlines())}
    base_scores = {r.sample_id: r for r in (
        Score.model_validate(v) for v in json.loads((RESULTS / "raw_score/scores.json").read_text()))}
    ours_scores = {r.sample_id: r for r in (
        Score.model_validate(v) for v in json.loads((RESULTS / "ours_score/scores.json").read_text()))}
    with np.load(RESULTS / "raw_replay/predictions.npz") as values:
        base = values["positions"]
    with np.load(RESULTS / "ours_selected/predictions.npz") as values:
        ours = values["positions"]
    with np.load(DATA / "inputs.npz") as values:
        np.testing.assert_array_equal(base, values["sol"])
        np.testing.assert_allclose(ours, .75 * values["sol"] + .25 * values["cv"], rtol=0, atol=1e-12)
    assert base.shape == ours.shape == (len(indices), 12, 2)
    cases = []
    for i, record in enumerate(indices):
        sid = record.sample_id
        if not np.isfinite(base[i]).all() or not np.isfinite(ours[i]).all():
            assert base_scores[sid].ade is None and ours_scores[sid].ade is None
            continue
        case = Case(sid, record.dataset, observations[sid], truths[sid], base[i], ours[i],
                    base_scores[sid], ours_scores[sid])
        assert len(case.primary_observed) == 9 and len(case.primary_truth) == 12
        assert all(p[1] == case.observation.primary_id for p in case.truth.paths[0])
        assert [p[0] for p in case.truth.paths[0]] == case.observation.future_frames
        assert case.observation.cutoff_s == 3.2 and case.observation.dt_s == .4
        np.testing.assert_allclose(case.observation.future_times_s, np.arange(9, 21) * .4, atol=1e-12)
        for prediction, score in ((case.baseline, case.baseline_score), (case.ours, case.ours_score)):
            errors = np.linalg.norm(prediction - case.primary_truth, axis=1)
            np.testing.assert_allclose([errors.mean(), errors[-1]], [score.ade, score.fde], atol=1e-12)
        cases.append(case)
    assert len(cases) == 1939 and len(indices) - len(cases) == 4
    return cases


def select_cases(cases: list[Case], datasets: frozenset[str], output: Path | None = None) -> list[Case]:
    """절대 오차·동일 인물·시간 간격을 함께 제한해 원영상별10개를 고른다."""
    from trajectory_selection import select
    return select([case for case in cases if case.dataset in datasets], "api", output)
