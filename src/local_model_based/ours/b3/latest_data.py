# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pydantic"]
# ///
# How to run: latest_entry.py에서 import
"""최신 보행 자료의 관측·정답·참조 경로를 구분한다."""
from pathlib import Path
import json
from typing import Final
import numpy as np
from pydantic import BaseModel, ConfigDict
from common import DATA, BoundaryError, sha
from features import Batch, load

LATEST: Final = DATA / "latest"


class Fold(BaseModel):
    model_config = ConfigDict(frozen=True)
    fold: int
    stems: list[str]
    epoch: int


class Asset(BaseModel):
    model_config = ConfigDict(frozen=True)
    source: str
    path: str
    sha256: str


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    method: str
    selection: str
    folds: list[Fold]
    files: list[Asset]
    source_unchanged: bool


def manifest() -> Manifest:
    return Manifest.model_validate_json((LATEST / "manifest.json").read_text())


def checked(relative: str, assets: Manifest) -> Path:
    original = LATEST / relative
    path = original.resolve()
    if not path.is_relative_to(LATEST.resolve()) or original.is_symlink():
        raise BoundaryError("invalid_latest_asset", relative)
    record = next(row for row in assets.files if row.path == relative)
    if sha(path) != record.sha256:
        raise BoundaryError("latest_asset_hash_mismatch", relative)
    return path


def observations(fold: Fold, assets: Manifest) -> tuple[Batch, list[str]]:
    batches, ids = [], []
    for stem in fold.stems:
        batches.append(load(checked(f"inputs/{stem}_inputs.npz", assets)))
        rows = json.loads(checked(f"indices/{stem}_index.json", assets).read_text())
        ids.extend(row["sample_id"] for row in rows)
    batch = Batch(**{key: np.concatenate([getattr(part, key) for part in batches])
                     for key in Batch.__dataclass_fields__})
    if len(set(ids)) != len(ids) or len(ids) != len(batch.ego):
        raise BoundaryError("invalid_latest_observation_order", str(fold.fold))
    return batch, ids
