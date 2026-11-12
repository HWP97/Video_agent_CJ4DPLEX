"""Observation-only, non-learned transformations of cached Sol trajectories."""
from dataclasses import dataclass
from typing import Literal
import numpy as np

@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    family: Literal['delta', 'axis', 'cv', 'kf']
    first: float
    second: float = 1.

    def tie_key(self) -> tuple:
        distance = abs(self.first - 1) + (abs(self.second - 1) if self.family == 'axis' else 0)
        family_order = {'delta': 0, 'axis': 1, 'cv': 2, 'kf': 2}
        return distance, family_order[self.family], self.name

CANDIDATES = tuple(
    [Candidate(f'delta_{a:g}', 'delta', a) for a in (0., .25, .5, .75, 1.)]
    + [Candidate(f'axis_{a:g}_{b:g}', 'axis', a, b) for a in (.75, 1., 1.25) for b in (.75, 1., 1.25)]
    + [Candidate(f'{anchor}_blend_{b:g}', anchor, b) for anchor in ('cv', 'kf') for b in (.25, .5, .75)]
)

@dataclass(frozen=True, slots=True)
class Inputs:
    sol: np.ndarray
    cv: np.ndarray
    kf: np.ndarray
    last: np.ndarray
    velocity: np.ndarray

    def take(self, ids: np.ndarray) -> 'Inputs':
        return Inputs(*(getattr(self, key)[ids] for key in self.__dataclass_fields__))


def observed_velocity(points: np.ndarray) -> np.ndarray:
    assert points.ndim == 2 and points.shape[1] == 3 and len(points) >= 3
    assert np.isfinite(points).all() and np.all(np.diff(points[:, 0]) > 0)
    last = points[-3:]
    design = np.column_stack((last[:, 0] - last[-1, 0], np.ones(3)))
    return np.linalg.lstsq(design, last[:, 1:], rcond=None)[0][0]


def transform(data: Inputs, candidate: Candidate) -> np.ndarray:
    valid = np.isfinite(data.sol).all(axis=(1, 2))
    assert data.sol.shape == data.cv.shape == data.kf.shape
    assert data.last.shape == data.velocity.shape == (len(valid), 2)
    match candidate.family:
        case 'delta':
            if candidate.first == 1:
                result = data.sol.copy()
            else:
                result = data.last[:, None] + candidate.first * (data.sol - data.last[:, None])
        case 'axis':
            if candidate.first == candidate.second == 1:
                result = data.sol.copy()
            else:
                speed = np.linalg.norm(data.velocity, axis=1)
                direction = data.velocity / np.maximum(speed[:, None], 1e-8)
                lateral = np.column_stack((-direction[:, 1], direction[:, 0]))
                delta = data.sol - data.last[:, None]
                parallel = np.sum(delta * direction[:, None], axis=-1)[..., None] * direction[:, None]
                perpendicular = np.sum(delta * lateral[:, None], axis=-1)[..., None] * lateral[:, None]
                scaled = data.last[:, None] + candidate.first * parallel + candidate.second * perpendicular
                result = np.where((speed > 1e-8)[:, None, None], scaled, data.sol)
        case 'cv' | 'kf':
            anchor = data.cv if candidate.family == 'cv' else data.kf
            if candidate.first == 1:
                result = data.sol.copy()
            elif candidate.first == 0:
                result = anchor.copy()
            else:
                result = (1 - candidate.first) * anchor + candidate.first * data.sol
        case unexpected:
            raise AssertionError(unexpected)
    result[~valid] = np.nan
    assert np.array_equal(np.isfinite(result).all((1, 2)), valid)
    return result
