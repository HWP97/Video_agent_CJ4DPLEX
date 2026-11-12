# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
# How to run: imported by features.py
"""관측 토큰화에 필요한 타입만 보존한 데이터 계약."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class Track:
    pedestrian: int
    times: FloatArray
    xy: FloatArray

@dataclass(frozen=True, slots=True)
class Observation:
    sample_id: str
    dataset: str
    primary: int
    frames: tuple[int, ...]
    xy: FloatArray
    tracks: tuple[Track, ...]
