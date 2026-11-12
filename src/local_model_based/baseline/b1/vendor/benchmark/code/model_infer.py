"""Local PGAGS inference; only observed images/annotations enter the model."""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from model_prefix import anticipate_prefix
from model_runtime import install_roi_compat

ROOT = Path(__file__).resolve().parents[5] / "data/b1/benchmark"
SOURCE = Path(__file__).resolve().parents[1] / "model/SceneSayer"
sys.path.insert(0, str(SOURCE))
install_roi_compat()
from dataloader.action_genome.ag_dataset import AG
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob
from lib.object_detector import Detector
from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher
from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
import lib.word_vectors


def load_models(mode):
    vocabulary = AG.__new__(AG)
    vocabulary._data_path = str(ROOT / "data")
    vocabulary._fetch_object_classes()
    vocabulary._fetch_relationship_classes()
    # All embedding parameters are restored by strict checkpoint loading below.
    lib.word_vectors.obj_edge_vectors = lambda names, wv_dim=200, **kw: torch.zeros(len(names), wv_dim)
    from lib.supervised.sga import blocks
    blocks.obj_edge_vectors = lib.word_vectors.obj_edge_vectors
    from lib.supervised.sga.scene_sayer_ode import SceneSayerODE
    model = SceneSayerODE(mode, 3, 6, 17, vocabulary.object_classes, max_window=3)
    filename = "ode_partial_40_sgcls_future_3_epoch_3.tar" if mode == "sgcls" else "ode_partial_70_predcls_future_3_epoch_3.tar"
    weights = ROOT.parent / "weights" / filename
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint["ode_state_dict"]
    # Historical checkpoints retain a separate classifier unused by pinned forward.
    if any(key.startswith("object_classifier.") for key in state):
        model.object_classifier = copy.deepcopy(model.dsgdetr.object_classifier)
    model.load_state_dict(state, strict=True)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    del checkpoint, state
    model = model.cuda().eval()
    detector = Detector(False, vocabulary.object_classes, False, mode).cuda().eval()
    return model, detector, HungarianMatcher(0.5, 1, 1, 0.5).eval()


def observed_entry(detector, matcher, case, observation, mode):
    annotations = copy.deepcopy(observation["frames"])
    processed, scales, hashes = [], [], []
    for name, frame in zip(case["observed_frame_ids"], annotations):
        path = ROOT / "data/frames" / name
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable observed image: {name}")
        image, scale = prep_im_for_blob(image, [[[102.9801, 115.9465, 122.7717]]], 600, 1000)
        processed.append(image)
        scales.append(scale)
        for obj in frame:
            if "person_bbox" in obj:
                obj["person_bbox"] = np.asarray(obj["person_bbox"], dtype=np.float32)
            else:
                obj["bbox"] = np.asarray(obj["bbox"], dtype=np.float32)
                for key in ["attention_relationship", "spatial_relationship", "contacting_relationship"]:
                    obj[key] = torch.tensor(obj[key], dtype=torch.long)
    if len(annotations) != case["observed_count"] or len(processed) != len(annotations):
        raise ValueError("Observed image and annotation lengths differ")
    blob = im_list_to_blob(processed)
    data = torch.from_numpy(blob).permute(0, 3, 1, 2).cuda()
    info = torch.tensor([[blob.shape[1], blob.shape[2], scales[0]]], dtype=torch.float32).repeat(len(processed), 1).cuda()
    boxes = torch.zeros(len(processed), 1, 5, device="cuda")
    counts = torch.zeros(len(processed), dtype=torch.int64, device="cuda")
    entry = detector(data, info, boxes, counts, annotations, im_all=None)
    get_sequence_with_tracking(entry, annotations, matcher, (info[0][:2] / info[0, 2]).cpu(), mode)
    for key in ["attention_gt", "spatial_gt", "contacting_gt"]:
        entry.pop(key, None)
    if mode == "sgcls":
        entry.pop("labels", None)
    return entry, hashes


def run_case(model, detector, matcher, case, obs, mode, manifest, out):
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        entry, hashes = observed_entry(detector, matcher, case, obs, mode)
        future = [int(Path(name).stem) for name in case["future_frame_ids"]]
        persistence = {}
        prediction = anticipate_prefix(model, entry, future, persistence)
    for value in prediction.values():
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite model prediction")
    torch.cuda.synchronize()
    out.mkdir(parents=True, exist_ok=True)
    for method, result in [("ode", prediction), ("persistence", persistence)]:
        target = out / f"{case['case_id']}.{method}.pt"
        torch.save({k: v.cpu() for k, v in result.items()}, str(target) + ".tmp")
        os.replace(str(target) + ".tmp", target)
        with (out / f"{method}.jsonl").open("a") as stream:
            stream.write(json.dumps({"case_id": case["case_id"], "prediction": {k: v.cpu().tolist() for k, v in result.items()}}) + "\n")
    video = Path(case["observed_frame_ids"][0]).parent.stem
    frames = json.loads((ROOT / "data/frame_records" / f"{video}.json").read_text())
    actual_index = int(entry["im_idx"].max())
    nominal_id = case["observed_frame_ids"][-1]
    actual_id = case["observed_frame_ids"][actual_index]
    pts = {f["frame_id"]: f["pts_s"] for f in frames["frames"]}
    record = {"case_id": case["case_id"], "mode": mode, "observed_count": len(hashes),
              "observed_sha256": hashes, "future_count": len(future), "strict_load": True,
              "nominal_cutoff": {"frame_id": nominal_id, "pts_s": pts[nominal_id]},
              "actual_cutoff": {"frame_id": actual_id, "pts_s": pts[actual_id]},
              "ode_time_units": "native_1_based_frame_index_not_seconds", "manifest": manifest,
              "elapsed_s": time.monotonic() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
              "shapes": {k: list(v.shape) for k, v in prediction.items()}, "finite": True}
    target = out / f"{case['case_id']}.json"
    Path(str(target) + ".tmp").write_text(json.dumps(record, indent=2) + "\n")
    os.replace(str(target) + ".tmp", target)
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", nargs="+", default=["00607.mp4_0.3"])
    parser.add_argument("--mode", choices=["sgcls", "predcls"], default="sgcls")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    os.chdir(SOURCE)
    cohort = "filtered_small" if args.mode == "sgcls" else "unfiltered"
    protocol = ROOT / "protocol"
    cases = [json.loads(line) for line in (protocol / f"schedule_{cohort}.jsonl").open()]
    observations = {obj["case_id"]: obj for obj in map(json.loads, (protocol / f"observations_full_{cohort}.jsonl").open())}
    assert {case["case_id"] for case in cases} == set(observations)
    model, detector, matcher = load_models(args.mode)
    out = ROOT / "model/predictions" / (("full_" if args.full else "") + args.mode)
    out.mkdir(parents=True, exist_ok=True)
    source_files = [*SOURCE.rglob("*.py"), *ROOT.glob("code/model*.py")]
    manifest = {"source_commit": "1d48739435e3f5e0fd9e8cc012668f31b28b187f", "compatibility_version": 1,
                "sources_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in sorted(source_files))).hexdigest(),
                "weights": json.loads((ROOT / "model/weights/verified.json").read_text()),
                "initial_embeddings": "zero_initialized_then_all_checkpoint_tensors_verified_equal"}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    selected = cases if args.full else [case for case in cases if case["case_id"] in args.case]
    while selected:
        waiting = []
        for case in selected:
            if (ROOT / "model/STOP_REQUESTED").exists():
                return
            case_id = case["case_id"]
            done = out / f"{case_id}.json"
            if done.exists() and (out / f"{case_id}.persistence.pt").exists():
                continue
            if (out / f"{case_id}.error.json").exists():
                continue
            if case["future_count"] == 0:
                (out / f"{case_id}.empty_future.json").write_text(json.dumps({"case_id": case_id, "reason": "official_empty_future"}))
                continue
            video = Path(case["observed_frame_ids"][0]).parent.stem
            if not (ROOT / "data/frame_records" / f"{video}.json").exists():
                waiting.append(case)
                continue
            try:
                run_case(model, detector, matcher, case, observations[case_id], args.mode, "run_manifest.json", out)
            except (RuntimeError, ValueError, KeyError, OSError, IndexError) as error:
                (out / f"{case_id}.error.json").write_text(json.dumps({"case_id": case_id, "error": str(error), "traceback": traceback.format_exc()}))
                print(traceback.format_exc(), flush=True)
                if not args.full:
                    raise
            torch.cuda.empty_cache()
        selected = waiting
        if selected:
            print(json.dumps({"waiting_for_frames": len(selected)}), flush=True)
            time.sleep(30)
    print(json.dumps({"finished_scope": len(cases) if args.full else len(args.case)}), flush=True)


if __name__ == "__main__":
    main()
