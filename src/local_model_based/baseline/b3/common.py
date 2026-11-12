# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
# How to run: imported by entry.py
"""분리본의 파일 저장 및 추론 경계."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import sys
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA: Final = ROOT / "data/b3"
JSON = str | int | float | bool | None | list["JSON"] | dict[str, "JSON"]

@dataclass(frozen=True, slots=True)
class BoundaryError(Exception):
    reason: str
    detail: str = ""
    def __str__(self) -> str:
        return f"{self.reason}: {self.detail}"

def sha(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()

def save(path: Path, value: JSON) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)

def guard(inference: bool) -> None:
    """미래 GT와 네트워크를 실제 파일 접근 경계에서 차단한다."""
    for key in tuple(os.environ):
        if "API_KEY" in key or key in ("OPENAI_KEY", "SOL_KEY"):
            del os.environ[key]
    def audit(event: str, args: tuple) -> None:
        if event.startswith("socket.") and event not in ("socket.__new__",):
            raise BoundaryError("network_blocked")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if inference and ("ground_truth" in path.parts or path.name.endswith("_labels.npz")):
                raise BoundaryError("future_GT_blocked", str(path))
    sys.addaudithook(audit)

