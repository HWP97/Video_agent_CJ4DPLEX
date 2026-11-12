"""실제 원영상 위에 예측·정답 위치를 표시한다."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from b3_data import Array, Case
from b3_projection import VideoSource, project, source_time
from errors import DemoError
from video import Frame, encode

FPS: Final = 25
SIZE: Final = (1600, 900)
INK: Final = "#182433"
MUTED: Final = "#566373"
BLUE: Final = "#2260D5"
RED: Final = "#CF3348"
ORANGE: Final = "#A56612"
FONT: Final = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@dataclass(frozen=True, slots=True)
class Moment:
    frame: Frame
    relative_s: float


@lru_cache(maxsize=8)
def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def visible_path(points: Array, times: Array, now: float) -> Array:
    """표시용 보간만 수행하고 원 평가 좌표는 변경하지 않는다."""
    if now < times[0] or now > times[-1] + .4:
        return np.empty((0, 2))
    history = points[times <= now]
    if now < times[-1] and len(history):
        next_index = len(history)
        fraction = (now - times[next_index - 1]) / (times[next_index] - times[next_index - 1])
        current = points[next_index - 1] + fraction * (points[next_index] - points[next_index - 1])
        history = np.vstack((history, current))
    return history


def render_frame(case: Case, moment: Moment) -> Frame:
    now = moment.relative_s
    raster = moment.frame.copy()
    height, width = raster.shape[:2]
    observed = visible_path(case.primary_observed, np.arange(9)*.4, min(now, 3.2))
    pixels = np.rint(project(observed, (width, height))).astype(np.int32)
    cv2.polylines(raster, [pixels], False, (255, 255, 255), 2, cv2.LINE_AA)
    if now <= 3.2:
        cv2.circle(raster, tuple(pixels[-1]), 7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        times = np.array([3.2, *case.observation.future_times_s])
        for coordinates, color in ((case.baseline, (18, 102, 165)), (case.ours, (213, 96, 34)), (case.primary_truth, (72, 51, 207))):
            points = visible_path(np.vstack((case.primary_observed[-1], coordinates)), times, now)
            projected = np.rint(project(points, (width, height))).astype(np.int32)
            cv2.polylines(raster, [projected], False, color, 2, cv2.LINE_AA)
            cv2.circle(raster, tuple(projected[-1]), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(raster, tuple(projected[-1]), 5, color, 2, cv2.LINE_AA)
    image = Image.new("RGB", SIZE, "#F4F6F8")
    draw = ImageDraw.Draw(image)
    draw.text((32, 22), "Trajectory Forecasting", font=font(31), fill=INK)
    draw.text((32, 69), case.sample_id, font=font(20), fill=MUTED)
    phase = "OBSERVED FRAMES" if now <= 3.2 else "FUTURE PREDICTION"
    draw.text((1218, 25), phase, font=font(23), fill=BLUE)
    speed = "1.0x" if now <= 3.2 else "0.5x"
    draw.text((1218, 66), f"t = {now:4.2f}s | playback {speed}", font=font(18), fill=MUTED)
    scale = min(1152/width, 700/height)
    frame_size = (round(width*scale), round(height*scale))
    left, top = 32+(1152-frame_size[0])//2, 122+(700-frame_size[1])//2
    image.paste(Image.fromarray(cv2.cvtColor(raster, cv2.COLOR_BGR2RGB)).resize(frame_size), (left, top))
    labels = (("Observed target", "#344453"), ("Baseline / raw Sol", ORANGE), ("Ours / Sol + CV", BLUE), ("Ground truth", RED))
    for index, (label, color) in enumerate(labels):
        y = 134+index*39
        draw.line((1218, y+13, 1244, y+13), fill=color, width=4)
        draw.text((1256, y), label, font=font(19), fill=color)
    draw.text((1218, 320), "Full-horizon evaluation", font=font(23), fill=INK)
    draw.text((1218, 360), "Baseline              Ours", font=font(18), fill=MUTED)
    for i, (name, a, b) in enumerate((("ADE", case.baseline_score.ade, case.ours_score.ade), ("FDE", case.baseline_score.fde, case.ours_score.fde))):
        label = f"{a:.3f} m            {b:.3f} m" if now > 3.2 else "--                        --"
        draw.text((1218, 402+i*80), name, font=font(18), fill=MUTED)
        draw.text((1218, 430+i*80), label, font=font(22), fill=INK)
    c0, c1 = case.baseline_score.collision_gt, case.ours_score.collision_gt
    collision = f"Col-II: {c0}  ->  {c1}" if now > 3.2 else "Col-II: --"
    draw.text((1218, 590), collision, font=font(21), fill=RED if now > 3.2 and c1 > c0 else INK)
    draw.text((1218, 627), "0 = no collision; 1 = collision", font=font(17), fill=MUTED)
    draw.text((1218, 670), "Original video + projected points", font=font(17), fill=INK)
    draw.text((1218, 703), "Interpolation is display-only.", font=font(17), fill=MUTED)
    draw.text((1218, 736), "Metrics use the 12 forecast points.", font=font(17), fill=MUTED)
    draw.rectangle((32, 847, 1568, 854), fill="#CBD2DA")
    draw.rectangle((32, 847, 32+int(1536*3.2/8), 854), fill=INK)
    x = 32+int(1536*now/8)
    draw.ellipse((x-5, 843, x+5, 859), fill=BLUE)
    draw.text((32, 865), "Observed: 9 points / 3.2s", font=font(17), fill=MUTED)
    draw.text((720, 865), "Future: 12 points / 4.8s | displayed at 0.5x", font=font(17), fill=MUTED)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def frames(case: Case, source: VideoSource) -> Iterator[Frame]:
    track = next(t for t in case.observation.tracks if t.pedestrian_id == case.observation.primary_id)
    start = track.points[0][0] + source.frame_offset
    capture = cv2.VideoCapture(str(source.path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        native = []
        for _ in range(201):
            ok, frame = capture.read()
            if not ok:
                raise DemoError(f"원영상의 필요한 프레임을 읽지 못했습니다: {case.sample_id}")
            native.append(frame)
        for display_index in range(321):
            now = source_time(display_index/FPS)
            native_index = min(200, int(np.floor(now*FPS+1e-8)))
            yield render_frame(case, Moment(native[native_index], now))
    finally:
        capture.release()


def write_video(case: Case, source: VideoSource, target: Path) -> None:
    encode(target, frames(case, source), FPS)
