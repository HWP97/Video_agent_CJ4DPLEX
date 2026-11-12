"""Crowds의 명목 meter 좌표를 원영상 중심 좌표로 되돌린다."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from b3_data import Array, ROOT
from errors import DemoError

OUTPUT: Final = ROOT / "demo/api_based/trajectory_forecasting"


class VideoSource(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset: str
    path: Path
    frame_offset: int
    width: int
    height: int
    alignment_evidence: Path


def load_sources() -> tuple[VideoSource, ...]:
    manifest = Path(os.environ.get("PEDESTRIAN_VIDEO_MANIFEST", str(OUTPUT / "source_videos.json")))
    parsed = tuple(VideoSource.model_validate(row) for row in json.loads(manifest.read_text()))
    video_root = os.environ.get("TRAJECTORY_VIDEO_ROOT")
    sources = tuple(source.model_copy(update={"path": Path(video_root) / source.path.name}) for source in parsed) if video_root else parsed
    for source in sources:
        if source.dataset not in {"crowds_students001", "crowds_students003", "crowds_zara01"}:
            raise DemoError(f"원영상 투영을 검증하지 않은 데이터입니다: {source.dataset}")
        if not source.path.is_file() or not source.alignment_evidence.is_file():
            raise DemoError(f"원영상 또는 투영 확인 자료가 없습니다: {source.dataset}")
        capture = cv2.VideoCapture(str(source.path))
        try:
            observed = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if observed != (source.width, source.height) or abs(capture.get(cv2.CAP_PROP_FPS) - 25.) > .001:
                raise DemoError(f"원영상 크기 또는25fps가 계약과 다릅니다: {source.dataset}")
        finally:
            capture.release()
    return sources


def project(points: Array, dimensions: tuple[int, int]) -> Array:
    """저자 변환의 x×0.0210, y×0.0239와 중심 원점·상향y축을 역변환한다."""
    width, height = dimensions
    return points * np.array([1 / .0210, -1 / .0239]) + np.array([width / 2, height / 2])


def source_time(display_time: float) -> float:
    """관측은1배속, cutoff 이후는0.5배속으로 같은 원영상 시각을 반환한다."""
    return display_time if display_time <= 3.2 else 3.2 + (display_time - 3.2) * .5
