"""소스와 결과의 제출 폴더 경계를 한 곳에서 정의한다."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
STRUCTURED = ROOT.parent.name == "src"
PROJECT = ROOT.parents[1] if STRUCTURED else ROOT.parent
WORKSPACE = PROJECT.parent if STRUCTURED else ROOT.parent
RESULTS = PROJECT / "results" / ROOT.name if STRUCTURED else ROOT / "results"
REPORTS = PROJECT.parent / "HWP_CJ4DPLEX_AI_support" / "reports" / ROOT.name if STRUCTURED else ROOT / "reports"


def artifact_path(value: str | Path) -> Path:
    """원 manifest의 results/ 상대경로를 이동된 결과 위치로 해석한다."""
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "results":
        return RESULTS.joinpath(*path.parts[1:])
    if path.parts and path.parts[0] == "reports":
        return REPORTS.joinpath(*path.parts[1:])
    if path.parts and path.parts[0] == "src":
        return ROOT.joinpath("ours", *path.parts[1:])
    return ROOT / path
