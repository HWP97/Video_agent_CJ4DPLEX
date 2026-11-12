# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pydantic"]
# ///
# How to run: main.py b3 ours evaluate --prediction <predictions.npz>
"""최신 동일 사례의 ADE/FDE 및 관측 이웃 충돌률을 평가한다."""
from pathlib import Path
import json
import sys
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from common import BoundaryError, save, sha
from latest_data import LATEST, checked, manifest


def fresh_collision(values: NDArray[np.float64], rows: list, raw_path: Path) -> NDArray[np.bool_]:
    """원 scorer의 endpoint/midpoint 연산으로 관측 이웃 충돌을 계산한다."""
    sys.path.insert(0, str(LATEST / "source"))
    from collision_metric import collision_segments
    with np.load(raw_path) as raw:
        tracks = raw["tracks"]
        scenes = {int(row[0]): int(row[1]) for row in raw["scenes"]}
    tracks = tracks[np.argsort(tracks[:, 0], kind="stable")]
    frames, first, counts = np.unique(tracks[:, 0], return_index=True, return_counts=True)
    by_frame = {int(frame): tracks[start:start + count] for frame, start, count in zip(frames, first, counts)}
    result = np.zeros((len(rows), 2), bool)
    for index, row in enumerate(rows):
        primary = scenes[int(row["sample_id"].rsplit("_", 1)[1])]
        visible = np.unique(np.concatenate([by_frame[frame][:, 1] for frame in row["observed_frames"]]))
        pieces = []
        for step, frame in enumerate(row["future_frames"]):
            neighbors = by_frame[frame]
            neighbors = neighbors[neighbors[:, 1] != primary]
            pieces.append(np.column_stack((np.full(len(neighbors), step), neighbors[:, 1:])))
        future = np.concatenate(pieces)
        future = future[np.argsort(future[:, 1], kind="stable")]
        paired = future[:-1, 1] == future[1:, 1]
        left, right = future[:-1][paired], future[1:][paired]
        hits = collision_segments(values[index:index + 1], left[:, 0].astype(int), right[:, 0].astype(int),
                                  left[:, 2:], right[:, 2:], np.isin(left[:, 1], visible))
        result[index] = [hit[0] for hit in hits]
    return result


def evaluate(prediction: Path, output: Path, stage: Literal["baseline", "ours"]) -> None:
    assets = manifest()
    method = assets.method if stage == "ours" else "baseline_attention"
    with np.load(prediction) as saved:
        positions, ids = saved["positions"], saved["sample_ids"].tolist()
    if positions.shape != (240890, 12, 2) or len(set(ids)) != 240890 or not np.isfinite(positions).all():
        raise BoundaryError("invalid_latest_predictions", str(prediction))
    total = np.zeros(4)
    offset, reuse_count, records = 0, 0, []
    checked("source/collision_metric.py", assets)
    for fold in assets.folds:
        reference = np.load(checked(f"reference/{stage}_fold{fold.fold}.npy", assets), mmap_mode="r")
        fold_offset = 0
        for stem in fold.stems:
            rows = json.loads(checked(f"indices/{stem}_index.json", assets).read_text())
            count = len(rows)
            assert ids[offset:offset + count] == [row["sample_id"] for row in rows]
            values = positions[offset:offset + count]
            with np.load(checked(f"ground_truth/{stem}_labels.npz", assets)) as labels:
                distance = np.linalg.norm(values - labels["xy"], axis=-1)
            ade, fde = distance.mean(-1), distance[:, -1]
            reused = np.array_equal(values, reference[fold_offset:fold_offset + count])
            if reused:
                with np.load(checked(f"ground_truth/scores/{stem}.npz", assets)) as scores:
                    index = scores["methods"].tolist().index(method)
                    assert scores["sample_ids"].tolist() == [row["sample_id"] for row in rows]
                    assert np.array_equal(ade, scores["ADE"][index])
                    assert np.array_equal(fde, scores["FDE"][index])
                    collision = scores["collision"][index]
                reuse_count += count
            else:
                collision = fresh_collision(values, rows, checked(f"ground_truth/raw/{stem}.npz", assets))
            with (output / (stem + ".npz")).open("xb") as stream:
                np.savez_compressed(stream, sample_ids=np.array(ids[offset:offset + count]), ADE=ade, FDE=fde, collision=collision)
            total += [ade.sum(), fde.sum(), collision[:, 0].sum(), collision[:, 1].sum()]
            offset += count
            fold_offset += count
            records.append({"stem": stem, "n": count, "collision_reused_exact_prediction": bool(reused)})
    assert offset == 240890
    save(output / "scores.json", {"method": method, "n": offset, "ADE": float(total[0] / offset),
        "FDE": float(total[1] / offset), "ColI_all": float(100 * total[2] / offset),
        "ColI_obs": float(100 * total[3] / offset), "units": "source coordinate units",
        "collision": "0.2 source-unit endpoint/midpoint; observed-neighbor GT, not predicted-neighbor Col-II",
        "prediction_sha256": sha(prediction), "collision_reused_samples": reuse_count, "files": records})
