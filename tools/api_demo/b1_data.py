# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "opencv-python-headless", "Pillow"]
# ///
# 실행: 기존 b1 환경에서 b1_demo.py가 가져온다.
"""b1 동결 비교 자료와 주석 시점을 영상 렌더러에 연결한다."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import sys
from typing import Final

import numpy as np
import torch

PROJECT: Final = Path(__file__).resolve().parents[2]
API: Final = PROJECT / "src/api_based"
DATA: Final = API / "data/b1"
SOURCE: Final = PROJECT.parent / "my_works/b1/benchmarks/action_genome_20261106/data"
sys.path[:0] = [str(API / "baseline/b1/vendor"), str(API / "ours/b1")]
from adapters import Entry
from adaptive import predict
from ours_method import decoder
from scoring import GroundTruth, evaluate, official

OBJECTS: Final = ["background"] + (DATA / "benchmark/data/annotations/object_classes.txt").read_text().splitlines()
RELATIONS: Final = ["looking at", "not looking at", "unsure", "above", "beneath", "in front of", "behind", "on the side of", "in", "carrying", "covered by", "drinking from", "eating", "have on back", "holding", "leaning on", "lying on", "not contacting", "other relation", "sitting on", "standing on", "touching", "twisting", "wearing", "wiping", "writing on"]


@dataclass(frozen=True, slots=True)
class Candidate:
    sample_id: str
    mode: str
    video_id: str
    baseline: float
    ours: float
    baseline_wc: float
    ours_wc: float

    @property
    def improvement(self) -> float:
        return self.ours - self.baseline


@dataclass(frozen=True, slots=True)
class Snapshot:
    time: float
    image: Path
    boxes: np.ndarray
    classes: np.ndarray
    baseline: tuple[str, ...]
    ours: tuple[str, ...]
    truth: tuple[str, ...]
    baseline_recall: float
    ours_recall: float
    baseline_matches: tuple[tuple[int, ...], ...]
    ours_matches: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class Scene:
    candidate: Candidate
    cutoff: float
    end: float
    native_pts: np.ndarray
    snapshots: tuple[Snapshot, ...]
    source_video: Path


def unique_ranked(candidates: list[Candidate], count: int, descending: bool) -> list[Candidate]:
    """Rank by paired improvement and retain one case per source video."""
    ranked = sorted(candidates, key=lambda c: ((-1 if descending else 1) * c.improvement, c.sample_id))
    selected: list[Candidate] = []
    seen: set[str] = set()
    for candidate in ranked:
        if candidate.video_id not in seen:
            selected.append(candidate)
            seen.add(candidate.video_id)
        if len(selected) == count:
            break
    return selected


def candidates() -> list[Candidate]:
    result: list[Candidate] = []
    for path in sorted((PROJECT / "results/api_based/b1/frozen").glob("results_full_*/per_case.jsonl")):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                item = row["schedule"]
                if item["ratio"] != 0.5 or "macro_direct" not in row["methods"]:
                    continue
                base, ours = row["methods"]["sol"], row["methods"]["macro_direct"]
                result.append(Candidate(item["sample_id"], item["mode"], item["video_id"],
                    base["no"]["R_sum"][1] / base["no"]["frames"],
                    ours["no"]["R_sum"][1] / ours["no"]["frames"],
                    base["with"]["R_sum"][1] / base["with"]["frames"],
                    ours["with"]["R_sum"][1] / ours["with"]["frames"]))
    return result


def label(subject: int, obj: int, predicate: int) -> str:
    return f"{OBJECTS[subject]} > {RELATIONS[predicate]} > {OBJECTS[obj]}"


def ranked_labels(entry: Entry) -> tuple[str, ...]:
    scores = entry.objects[entry.pairs].prod(1)[:, None] * entry.relations
    indices = official.argsort_desc(scores)[:20]
    return tuple(label(int(entry.classes[entry.pairs[row, 0]]),
                       int(entry.classes[entry.pairs[row, 1]]), int(predicate))
                 for row, predicate in indices)


def relation_matches(entry: Entry, truth: GroundTruth, mode: str) -> tuple[tuple[int, ...], ...]:
    """Use the unchanged official NC matcher, including class, direction and IoU=0."""
    metrics = {mode + "_recall": {20: []},
               mode + "_mean_recall_collect": {20: [[] for _ in range(26)]}}
    matches, _, _ = official.evaluate_from_dict(truth, entry.official(), mode, metrics,
        method="no", threshold=None, num_rel=26, iou_thresh=0.0)
    return tuple(tuple(int(index) for index in matched) for matched in matches[:20])


def load_scene(candidate: Candidate) -> Scene:
    """Restore fixed predictions before consulting GT for the display and scoring."""
    protocol = "filtered_small" if candidate.mode == "sgcls" else "unfiltered"
    with (DATA / f"benchmark/protocol/{protocol}/cases.jsonl").open() as stream:
        case = next(json.loads(line) for line in stream if json.loads(line)["case_id"] == candidate.sample_id.split(":", 1)[1])
    timeline = json.loads((SOURCE / "frame_records" / (Path(candidate.video_id).stem + ".json")).read_text())
    times = {row["frame_id"]: row["pts_s"] for row in timeline["frames"]}
    with (DATA / "prepared" / (candidate.sample_id.replace(":", "__") + ".pkl")).open("rb") as stream:
        record = pickle.load(stream)
    torch.set_num_threads(1)
    model = decoder(candidate.mode, "macro_direct")
    entries = tuple(predict(frame.entry, feature, model)
                    for frame, feature in zip(record["frames"]["no"], record["features"]["no"], strict=True))
    snapshots: list[Snapshot] = []
    for frame_id, frame, ours in zip(case["future_frame_ids"], record["frames"]["no"], entries, strict=True):
        base_score = evaluate(frame.entry, frame.truth, candidate.mode, "no")
        ours_score = evaluate(ours, frame.truth, candidate.mode, "no")
        truth = frame.truth
        count = len(truth["gt_relations"])
        base_matches = relation_matches(frame.entry, truth, candidate.mode)
        ours_matches = relation_matches(ours, truth, candidate.mode)
        assert set(i for group in base_matches for i in group) == set(base_score.hit_sets[1])
        assert set(i for group in ours_matches for i in group) == set(ours_score.hit_sets[1])
        snapshots.append(Snapshot(float(times[frame_id]), SOURCE / "frames" / frame_id,
            truth["gt_boxes"], truth["gt_classes"], ranked_labels(frame.entry), ranked_labels(ours),
            tuple(label(int(truth["gt_classes"][s]), int(truth["gt_classes"][o]), int(p))
                  for s, o, p in truth["gt_relations"]),
            len(base_score.hit_sets[1]) / count, len(ours_score.hit_sets[1]) / count,
            base_matches, ours_matches))
    np.testing.assert_allclose(np.mean([s.baseline_recall for s in snapshots]), candidate.baseline, atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.mean([s.ours_recall for s in snapshots]), candidate.ours, atol=1e-10, rtol=0)
    return Scene(candidate, float(times[case["observed_frame_ids"][-1]]), float(times[case["future_frame_ids"][-1]]),
                 np.asarray(timeline["native_frame_pts_s"]), tuple(snapshots), SOURCE / "videos" / candidate.video_id)
