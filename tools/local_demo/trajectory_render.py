# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "opencv-python-headless", "pydantic"]
# ///
# How to run: tools/local_demo/trajectories.py에서 import
"""검증된 원영상 투영 렌더러에 로컬 모델 이름과 지표 정의를 표시한다."""
from collections.abc import Iterator
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from trajectory_data import Case
from b3_projection import VideoSource
from b3_render import frames, font, INK, MUTED, BLUE, RED, ORANGE
from video import Frame, encode


def local_frames(case: Case, source: VideoSource) -> Iterator[Frame]:
    for index, frame in enumerate(frames(case, source)):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        draw.rectangle((1205, 118, 1599, 835), fill="#F4F6F8")
        draw.text((1218, 120), "LOCAL MODELS", font=font(23), fill=INK)
        labels = (("Observed target", INK), ("Baseline / attention", ORANGE), ("Ours / D04 two-state", BLUE), ("Ground truth", RED))
        for i, (label, color) in enumerate(labels):
            y = 165 + i * 39
            draw.line((1218, y + 13, 1244, y + 13), fill=color, width=4)
            draw.text((1256, y), label, font=font(18), fill=color)
        draw.text((1218, 340), "Full-horizon evaluation", font=font(22), fill=INK)
        draw.text((1218, 380), "Baseline              Ours", font=font(18), fill=MUTED)
        for i, (name, first, last) in enumerate((("ADE", case.baseline_score.ade, case.ours_score.ade), ("FDE", case.baseline_score.fde, case.ours_score.fde))):
            draw.text((1218, 420 + i * 75), name, font=font(18), fill=MUTED)
            value = f"{first:.3f}                 {last:.3f}" if index > 80 else "--                        --"
            draw.text((1218, 447 + i * 75), value, font=font(22), fill=INK)
        draw.text((1218, 579), "Distance: source coordinate units", font=font(16), fill=MUTED)
        collision = f"{case.baseline_score.collision_gt}  ->  {case.ours_score.collision_gt}" if index > 80 else "--"
        draw.text((1218, 620), "Observed-neighbor GT collision", font=font(17), fill=INK)
        draw.text((1218, 652), collision, font=font(23), fill=INK)
        draw.text((1218, 692), "No API / frozen local models", font=font(18), fill=INK)
        draw.text((1218, 730), "Original video + projected points", font=font(17), fill=MUTED)
        draw.text((1218, 762), "Interpolation is display-only.", font=font(17), fill=MUTED)
        yield cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def write(case: Case, source: VideoSource, target: Path) -> None:
    encode(target, local_frames(case, source), 25)
