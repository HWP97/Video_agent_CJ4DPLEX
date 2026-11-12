# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "Pillow"]
# ///
# 실행: b1_demo.py가 호출한다.
"""연속 원영상과 정확한 주석 스냅샷의 관계 비교 MP4를 만든다."""
from __future__ import annotations

import math
from pathlib import Path
import subprocess
from typing import Final

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from b1_data import OBJECTS, Scene, Snapshot

SIZE: Final = (1600, 900)
FPS: Final = 15
INK: Final = (229, 235, 244)
MUTED: Final = (167, 181, 198)
BLUE: Final = (103, 164, 255)
RED: Final = (255, 127, 131)
LIME: Final = (181, 239, 111)
FONTS: Final = {n: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", n) for n in (17, 19, 21, 25, 30)}


def fitted(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    output = Image.new("RGB", size, (7, 13, 23))
    source = image.copy()
    source.thumbnail(size)
    output.paste(source, ((size[0] - source.width) // 2, (size[1] - source.height) // 2))
    return output


def annotated(snapshot: Snapshot) -> Image.Image:
    with Image.open(snapshot.image) as source:
        original = source.convert("RGB")
    draw = ImageDraw.Draw(original)
    for box, cls in zip(snapshot.boxes, snapshot.classes, strict=True):
        x0, y0, x1, y1 = map(float, box)
        draw.rectangle((x0, y0, x1, y1), outline=RED, width=2)
        draw.text((x0 + 2, max(0, y0 - 19)), OBJECTS[int(cls)], fill=RED, font=FONTS[17], stroke_width=1, stroke_fill=(0, 0, 0))
    return fitted(original, (340, 250))


def relation_column(draw: ImageDraw.ImageDraw, rows: tuple[tuple[str, bool], ...], origin: tuple[int, int]) -> None:
    x, y = origin
    for number, (line, matched) in enumerate(rows[:20], 1):
        draw.text((x, y + (number - 1) * 25), f"{number:02d} {line}", fill=LIME if matched else INK, font=FONTS[17])


def source_time(playback_time: float, cutoff: float) -> float:
    """Keep observations at 1x and map future playback to half-speed source time."""
    return min(playback_time, cutoff) + max(playback_time - cutoff, 0) * 0.5


def canvas(scene: Scene, rgb: Image.Image, time: float) -> Image.Image:
    image = Image.new("RGB", SIZE, (12, 20, 33))
    draw = ImageDraw.Draw(image)
    candidate = scene.candidate
    observed = time <= scene.cutoff
    phase = "OBSERVED FRAMES / 1x" if observed else "FUTURE GT VIDEO / 0.5x"
    draw.text((24, 18), f"Scene Graph Anticipation / {candidate.sample_id} / {phase}", font=FONTS[25], fill=INK)
    draw.text((24, 62), f"Source t = {time:.2f}s     cutoff = {scene.cutoff:.2f}s     observation ratio = 0.50", font=FONTS[21], fill=MUTED)
    image.paste(fitted(rgb, (730, 430)), (24, 104))
    draw.text((24, 546), "Continuous source video / native PTS", font=FONTS[19], fill=MUTED)
    draw.text((790, 109), "BASELINE / raw Sol", font=FONTS[25], fill=MUTED)
    draw.text((1190, 109), "OURS / macro_direct", font=FONTS[25], fill=BLUE)
    draw.text((790, 147), "Top 20 NC relations", font=FONTS[19], fill=MUTED)
    draw.text((1190, 147), "Top 20 NC relations", font=FONTS[19], fill=MUTED)
    draw.text((790, 172), "Lime = official GT match (NC, IoU=0)", font=FONTS[17], fill=LIME)
    if observed:
        draw.text((790, 220), "Observations only", font=FONTS[25], fill=INK)
        draw.text((790, 267), "Future GT is not an inference input.", font=FONTS[21], fill=MUTED)
        draw.text((790, 310), "Prediction display starts after cutoff.", font=FONTS[21], fill=MUTED)
    else:
        snapshot = min(scene.snapshots, key=lambda snap: abs(snap.time - time))
        image.paste(annotated(snapshot), (24, 592))
        draw.text((24, 852), f"Scored frame @ {snapshot.time:.2f}s / GT boxes", font=FONTS[17], fill=RED)
        draw.text((384, 588), "GT RELATIONS / evaluation only", font=FONTS[19], fill=RED)
        for number, line in enumerate(snapshot.truth[:11]):
            draw.text((384, 619 + number * 21), line, font=FONTS[17], fill=INK)
        if len(snapshot.truth) > 11:
            draw.text((384, 850), f"+ {len(snapshot.truth) - 11} GT relations in evidence JSON", font=FONTS[17], fill=MUTED)
        relation_column(draw, tuple(zip(snapshot.baseline, map(bool, snapshot.baseline_matches), strict=True)), (790, 192))
        relation_column(draw, tuple(zip(snapshot.ours, map(bool, snapshot.ours_matches), strict=True)), (1190, 192))
        draw.text((790, 711), f"Annotation @ {snapshot.time:.2f}s (nearest scored frame)", font=FONTS[19], fill=MUTED)
        draw.text((790, 746), f"Frame R@20: {snapshot.baseline_recall * 100:.1f}%  ->  {snapshot.ours_recall * 100:.1f}%", font=FONTS[21], fill=INK)
    draw.text((790, 795), f"Case R@20: {candidate.baseline * 100:.1f}%  ->  {candidate.ours * 100:.1f}%", font=FONTS[25], fill=BLUE)
    draw.text((790, 837), f"WC R@20: {candidate.baseline_wc * 100:.1f}% -> {candidate.ours_wc * 100:.1f}%", font=FONTS[19], fill=MUTED)
    draw.text((790, 868), "Frozen cached prediction / GT positions are not predicted", font=FONTS[17], fill=MUTED)
    return image


def render(scene: Scene, destination: Path) -> None:
    """Render a half-speed future while preserving original PTS and score values."""
    temporary = destination.with_suffix(".rendering.mp4")
    command = ["/usr/bin/ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1600x900", "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    capture = cv2.VideoCapture(str(scene.source_video))
    if not capture.isOpened():
        raise FileNotFoundError(scene.source_video)
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            if encoder.stdin is None:
                raise BrokenPipeError(destination)
            decoded_index = -1
            frame = np.empty((1, 1, 3), dtype=np.uint8)
            poster_saved = False
            duration = scene.cutoff + 2 * (scene.end - scene.cutoff)
            for frame_number in range(math.floor(duration * FPS) + 1):
                time = source_time(frame_number / FPS, scene.cutoff)
                target = max(0, int(np.searchsorted(scene.native_pts, time, side="right")) - 1)
                while decoded_index < target:
                    valid, frame = capture.read()
                    if not valid:
                        raise EOFError(scene.source_video)
                    decoded_index += 1
                rendered = canvas(scene, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), time)
                encoder.stdin.write(rendered.tobytes())
                if not poster_saved and time >= scene.cutoff + (scene.end - scene.cutoff) / 2:
                    rendered.save(destination.with_suffix(".rendering.jpg"), quality=92)
                    poster_saved = True
            encoder.stdin.close()
            code = encoder.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        temporary.replace(destination)
        destination.with_suffix(".rendering.jpg").replace(destination.with_suffix(".jpg"))
    finally:
        capture.release()
