# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless"]
# ///
# 실행: tools/api_demo 렌더러에서 가져옴
"""CPU 프레임을 웹 호환 H.264 영상으로 저장한다."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import os
import subprocess
from uuid import uuid4
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray
from errors import DemoError

Frame = NDArray[np.uint8]
INK: Final = (232, 236, 243)
BLUE: Final = (245, 166, 77)
RED: Final = (105, 105, 250)
GRAY: Final = (166, 178, 195)


def label(frame: Frame, content: str, location: tuple[int, int], color: tuple[int, int, int] = INK) -> None:
    cv2.putText(frame, content, location, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def encode(path: Path, frames: Iterable[Frame], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(frames)
    first = next(iterator)
    height, width = first.shape[:2]
    temporary = path.with_name(path.stem + "." + uuid4().hex + path.suffix)
    command = [os.environ.get("FFMPEG_BINARY", "/usr/bin/ffmpeg"), "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
               "-vcodec", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-threads", "2",
               "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        if process.stdin is None:
            raise DemoError("FFmpeg 입력 파이프를 열지 못했습니다")
        process.stdin.write(first.tobytes())
        for frame in iterator:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    temporary.replace(path)
