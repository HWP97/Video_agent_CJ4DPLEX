# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pydantic", "typing-extensions"]
# ///
# 실행: src/local_model_based/main.py b2 ours predict
"""C01 baseline과 D04를 같은 관측2740개에서 계산한다. hold는 추가 비교군이다."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Final, Literal
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from typing_extensions import assert_never

from bundle_paths import RESULTS, artifact_path

DATA: Final = Path(__file__).resolve().parents[2] / "data/b2/latest"
Array = NDArray[np.float64]
Method = Literal["B2-D04", "B2-C01", "baseline_hold"]


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True)
    sample_id: str
    split: str
    cutoff_s: float
    frame_pts_s: list[float]
    horizons_s: list[float]


@dataclass(frozen=True, slots=True)
class InvalidAsset(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def observations() -> list[Observation]:
    rows = [Observation.model_validate(row) for row in json.loads((DATA / "manifest.json").read_text())]
    if len(rows) != 2740 or len({r.sample_id for r in rows}) != len(rows):
        raise InvalidAsset("missing_or_duplicate_observation")
    if any(len(r.frame_pts_s) != 15 or max(r.frame_pts_s) > r.cutoff_s + 1e-9 or r.horizons_s != [.5, 1., 2., 3.] for r in rows):
        raise InvalidAsset("invalid_observation_protocol")
    return rows


def predict(method: Method) -> tuple[Array | None, Array, bool]:
    """관측 전용 행렬 계산을 실행하고 GT와 저장 예측 읽기를 차단한다."""
    active = True

    def guard(event: str, arguments: tuple) -> None:
        if active and event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(arguments[0])).resolve()
            if path.is_relative_to(DATA / "ground_truth") or path.is_relative_to(DATA / "reference"):
                raise PermissionError("prediction_target_access_forbidden")

    sys.addaudithook(guard)
    try:
        blocked = False
        try:
            (DATA / "ground_truth/labels_eval.json").read_bytes()
        except PermissionError as error:
            if str(error) != "prediction_target_access_forbidden":
                raise
            blocked = True
        assert blocked
        rows = observations()
        with np.load(DATA / "inputs.npz", allow_pickle=False) as saved:
            hold = saved["recognition"][:, -1].astype(np.float64)
            match method:
                case "baseline_hold":
                    return hold, np.repeat(hold[:, None], 4, axis=1), blocked
                case "B2-C01":
                    from c01_inference import predict_c01
                    prediction = predict_c01(saved["x"], saved["numeric"], hold,
                        np.array([r.frame_pts_s for r in rows], dtype=np.float32))
                    if prediction.shape != (2740, 4, 2) or not np.isfinite(prediction).all():
                        raise InvalidAsset("invalid_C01_prediction")
                    return None, prediction, blocked
                case "B2-D04":
                    raw, numeric = saved["x"], saved["numeric"]
                case unreachable:
                    assert_never(unreachable)
        visual = np.concatenate((raw[:, -1], raw.mean(1)), 1).astype(np.float64)
        with np.load(DATA / "model/transform.npz", allow_pickle=False) as t:
            projected = ((visual - t["mean"]) / t["scale"]) @ t["basis"]
            joined = np.concatenate((projected, numeric), 1)
            x = np.column_stack(((joined - t["fm"]) / t["fs"], np.ones(len(joined))))
        with np.load(DATA / "model/B2-D02.npz", allow_pickle=False) as parent:
            current = hold + x @ parent["coefficient"]
        with np.load(DATA / "model/B2-D04.npz", allow_pickle=False) as future:
            prediction = current[:, None] + (x @ future["coefficient"]).reshape(-1, 4, 2)
        if not np.isfinite(prediction).all() or prediction.shape != (2740, 4, 2):
            raise InvalidAsset("invalid_prediction")
        return current, prediction, blocked
    finally:
        active = False


def metric(prediction: Array, truth: Array) -> tuple[float, float]:
    p = prediction.clip(-1, 1).reshape(len(prediction), -1).astype(np.float64)
    y = truth.reshape(len(truth), -1).astype(np.float64)
    covariance = ((p - p.mean(0)) * (y - y.mean(0))).mean(0)
    denominator = p.var(0) + y.var(0) + (p.mean(0) - y.mean(0)) ** 2
    return float(np.abs(p - y).mean()), float((2 * covariance / np.maximum(denominator, 1e-15)).mean())


def main(default_method: Method = "B2-D04") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "predict", "infer", "replay", "evaluate", "verify"))
    parser.add_argument("--method", choices=("B2-D04", "B2-C01", "baseline_hold"), default=default_method)
    parser.add_argument("--sample-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = artifact_path(args.output) if args.output else RESULTS / "b2" / f"latest-{args.method}-{args.action}-{uuid4().hex}"
    destination = destination.resolve()
    if not destination.is_relative_to((RESULTS / "b2").resolve()):
        raise InvalidAsset("output_must_be_inside_family_results_b2")
    rows = observations()
    selected = [i for i, row in enumerate(rows) if args.sample_id is None or row.sample_id == args.sample_id]
    if not selected:
        raise InvalidAsset("unknown_sample_id")
    if args.sample_id is not None and args.action in ("evaluate", "verify"):
        parser.error("evaluate/verify는 전체 분모를 보존합니다")
    result = {"method": args.method, "samples": len(rows), "training": False, "paid_calls": 0,
              "input_source": "stored_observation_features", "new_video_forward": False}
    match args.action:
        case "status":
            result["selection"] = {"B2-D04": "preselected_before_official_test", "B2-C01": "posthoc_primary_baseline_designation", "baseline_hold": "supporting_persistence_comparator"}[args.method]
        case "predict" | "infer" | "replay":
            current, prediction, blocked = predict(args.method)
            result.update(negative_GT_open_blocked=blocked, GT_read=False,
                          predictions=[{"sample_id": rows[i].sample_id, "current": None if current is None else current[i].tolist(), "prediction": prediction[i].tolist()} for i in selected])
        case "evaluate":
            current, prediction, blocked = predict(args.method)
            labels = {}
            for name in ("train", "dev", "eval"):
                labels.update({row["sample_id"]: row for row in json.loads((DATA / f"ground_truth/labels_{name}.json").read_text())})
            truth = np.array([labels[r.sample_id]["future"] for r in rows], dtype=np.float64)
            now = np.array([labels[r.sample_id]["current"] for r in rows], dtype=np.float64)
            scores = []
            for split in ("train", "dev", "official_test", "additional"):
                mask = np.array([r.split == split for r in rows])
                mae, ccc = metric(prediction[mask], truth[mask])
                current_mae, current_ccc = (None, None) if current is None else metric(current[mask], now[mask])
                scores.append({"split": split, "n": int(mask.sum()), "MAE": mae, "CCC": ccc, "current_MAE": current_mae, "current_CCC": current_ccc})
            result.update(scores=scores, inference_GT_read=False, negative_GT_open_blocked=blocked)
        case "verify":
            records = json.loads((DATA / "assets.json").read_text())
            for record in records:
                path = DATA / record["destination"]
                if path.is_symlink() or digest(path) != record["sha256"]:
                    raise InvalidAsset("asset_hash_mismatch:" + record["destination"])
            c01_records = json.loads((DATA / "assets_c01.json").read_text())
            for record in c01_records:
                path = DATA.parents[2] / record["destination"]
                if path.is_symlink() or digest(path) != record["sha256"]:
                    raise InvalidAsset("asset_hash_mismatch:" + record["destination"])
            current, prediction, blocked = predict(args.method)
            match args.method:
                case "B2-D04" | "B2-C01":
                    with np.load(DATA / f"reference/{args.method}_prediction.npz", allow_pickle=False) as reference:
                        difference = float(np.abs(prediction - reference["prediction"]).max())
                        if current is not None:
                            difference = max(difference, float(np.abs(current - reference["current"]).max()))
                case "baseline_hold":
                    with np.load(DATA / "inputs.npz", allow_pickle=False) as reference:
                        expected = np.repeat(reference["recognition"][:, -1:, :], 4, axis=1)
                        difference = float(np.abs(prediction - expected).max())
                case unreachable:
                    assert_never(unreachable)
            if difference >= 1e-12:
                raise InvalidAsset("frozen_prediction_mismatch")
            result.update(verified_files=len(records)+len(c01_records), verified_prediction_method=args.method, full_replay_maxdiff=difference, negative_GT_open_blocked=blocked)
        case unreachable:
            assert_never(unreachable)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "result.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({"artifact": str(path), **({k: v for k, v in result.items() if k != "predictions"})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
