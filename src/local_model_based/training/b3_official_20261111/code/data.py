from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Track:
    pedestrian: int
    times: np.ndarray
    xy: np.ndarray


@dataclass(frozen=True)
class Observation:
    sample_id: str
    dataset: str
    primary: int
    frames: tuple
    xy: np.ndarray
    tracks: tuple

