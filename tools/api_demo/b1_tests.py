# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest", "numpy", "torch"]
# ///
# 실행: 기존 b1 Python -m pytest tools/api_demo/b1_tests.py
"""사례 선택과 관계 방향의 사용자 가시 계약을 검사한다."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from b1_data import Candidate, Entry, label, relation_matches, unique_ranked
from b1_render import LIME, relation_column, source_time


def test_selects_best_mode_once_per_video() -> None:
    cases = [Candidate("a", "sgcls", "same", 0, .8, 0, 0),
             Candidate("b", "predcls", "same", 0, .7, 0, 0),
             Candidate("c", "sgcls", "other", 0, .6, 0, 0)]
    chosen = unique_ranked(list(reversed(cases)), 2, True)
    assert [row.sample_id for row in chosen] == ["a", "c"]


def test_keeps_object_to_person_direction_for_spatial_relation() -> None:
    assert label(2, 1, 3) == "bag > above > person"


def test_worst_cases_are_sorted_by_negative_improvement() -> None:
    cases = [Candidate("a", "sgcls", "one", .8, .7, 0, 0),
             Candidate("b", "sgcls", "two", .8, .2, 0, 0)]
    assert [row.sample_id for row in unique_ranked(cases, 1, False)] == ["b"]


@pytest.mark.parametrize("object_class,expected", [(2, True), (3, False)])
def test_official_match_checks_object_class_and_relation_direction(object_class: int, expected: bool) -> None:
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    scores = np.zeros((3, 26))
    scores[0, 0], scores[1, 3], scores[2, 14] = .9, .8, .7
    entry = Entry(boxes, np.array([1, object_class]), np.ones(2),
                  np.array([[0, 1], [1, 0], [0, 1]]), scores, np.array([-1, 0]))
    truth = {"gt_classes": np.array([1, 2]), "gt_boxes": boxes,
             "gt_relations": np.array([[0, 1, 0], [0, 1, 3]])}
    matches = relation_matches(entry, truth, "sgcls")
    assert bool(matches[0]) is expected
    assert matches[1] == ()
    assert matches[2] == ()


def test_only_matched_relation_rows_are_lime() -> None:
    image = Image.new("RGB", (400, 60))
    relation_column(ImageDraw.Draw(image), (("person > looking at > bag", True),
                                           ("bag > above > person", False)), (0, 0))
    pixels = np.asarray(image)
    assert np.any(np.all(pixels[:25] == LIME, axis=2))
    assert not np.any(np.all(pixels[25:] == LIME, axis=2))


@pytest.mark.parametrize("playback,expected", [(0, 0), (4, 4), (8, 8), (10, 9), (20, 14)])
def test_only_future_playback_runs_at_half_speed(playback: float, expected: float) -> None:
    assert source_time(playback, 8) == expected
