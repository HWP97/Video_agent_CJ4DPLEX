"""데모 선별의 실제 원본 그룹 분리 계약을 검증한다."""
from b2_data import select_cases
from b2_tracking import TargetTracker
import numpy as np


def test_demo_selection_uses_twenty_distinct_original_groups() -> None:
    cases = select_cases()
    assert len(cases) == len({case.group for case in cases}) == 20
    assert all(case.metric_delta > 0 for case in cases)


def test_target_roi_follows_image_translation() -> None:
    image = np.random.default_rng(9).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    tracker = TargetTracker(image, [0.2, 0.2, 0.4, 0.4])
    moved = np.roll(image, (4, 6), axis=(0, 1))
    box = tracker.update(moved)
    assert box is not None
    assert abs(box[0] - 31) <= 1 and abs(box[1] - 29) <= 1


def test_target_roi_stops_on_scene_cut() -> None:
    image = np.random.default_rng(9).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    tracker = TargetTracker(image, [0.2, 0.2, 0.4, 0.4])
    assert tracker.update(np.zeros_like(image)) is None
