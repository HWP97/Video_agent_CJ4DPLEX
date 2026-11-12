# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless"]
# ///
# 실행: b2 렌더러에서 가져옴
"""관측 종료의 지정 ROI를 광학 흐름으로 따라가는 표시 전용 추적."""
from __future__ import annotations
import cv2
import numpy as np
from video import Frame


class TargetTracker:
    """프레임 간 점 위치와 ROI를 갱신하는 추적 상태."""
    def __init__(self, image: Frame, normalized: list[float]) -> None:
        height, width = image.shape[:2]
        self.box = np.array(normalized) * (width, height, width, height)
        x, y, w, h = self.box.astype(int)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[max(0,y):min(height,y+h), max(0,x):min(width,x+w)] = 255
        self.previous = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.points = cv2.goodFeaturesToTrack(self.previous, 60, 0.02, 3, mask=mask)
        self.active = self.points is not None and len(self.points) >= 5

    def update(self, image: Frame) -> tuple[int, int, int, int] | None:
        if not self.active:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        before = cv2.calcHist([self.previous], [0], None, [32], [0, 256])
        after = cv2.calcHist([gray], [0], None, [32], [0, 256])
        if cv2.compareHist(before/before.sum(), after/after.sum(), cv2.HISTCMP_BHATTACHARYYA) > 0.55:
            self.active = False
            return None
        moved, status, error = cv2.calcOpticalFlowPyrLK(self.previous, gray, self.points, None)
        if moved is None or status is None or error is None:
            self.active = False
            return None
        valid = (status.reshape(-1) == 1) & (error.reshape(-1) < 25)
        if int(valid.sum()) < 5:
            self.active = False
            return None
        delta = np.median((moved - self.points)[valid, 0], axis=0)
        if float(np.linalg.norm(delta)) > 80:
            self.active = False
            return None
        self.box[:2] += delta
        self.previous = gray
        self.points = moved[valid]
        x, y, w, h = self.box.astype(int)
        return int(x), int(y), int(w), int(h)
