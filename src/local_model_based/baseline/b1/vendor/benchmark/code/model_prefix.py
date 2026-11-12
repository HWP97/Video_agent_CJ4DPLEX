"""SceneSayer ODE anticipation from an observed-only detector/tracker entry."""
import torch
from torchdiffeq import odeint_adjoint as odeint


@torch.inference_mode()
def anticipate_prefix(model, entry: dict, future_frame_idx: list[int], persistence_out: dict | None = None) -> dict:
    """Future inputs are timestamp/frame indices only; no future objects or labels."""
    if "gt_annotation" in entry:
        raise ValueError("Remove annotations before passing the observed feature entry")
    observed_times = list(entry["frame_idx"])
    if not observed_times or not future_frame_idx:
        raise ValueError("Both observed and future frames are required")
    if any(b <= a for a, b in zip(observed_times + future_frame_idx,
                                  (observed_times + future_frame_idx)[1:])):
        raise ValueError("Frame indices must be strictly increasing across the cutoff")
    if entry["boxes"].numel() == 0 or entry["boxes"][:, 0].max() >= len(observed_times):
        raise ValueError("Boxes must belong only to observed frames")
    entry = model.dsgdetr(entry)
    im_idx = entry["im_idx"]
    if im_idx.numel() == 0:
        return {}
    last_frame = int(im_idx.max().item())
    mask = im_idx == last_frame
    pairs = entry["pair_idx"][mask]
    first_box, last_box = int(pairs.min()), int(pairs.max())
    object_count = last_box - first_box + 1
    times = torch.tensor([observed_times[last_frame], *future_frame_idx], dtype=torch.float32)
    latent = odeint(model.diff_func, entry["global_output"][mask], times,
                    method="explicit_adams", options={"max_order": 4, "step_size": 1})[1:]
    result = {
        "attention_distribution": model.dsgdetr.a_rel_compress(latent).flatten(0, 1),
        "spatial_distribution": model.dsgdetr.s_rel_compress(latent).sigmoid().flatten(0, 1),
        "contacting_distribution": model.dsgdetr.c_rel_compress(latent).sigmoid().flatten(0, 1),
    }
    score_key, label_key = ("scores", "labels") if model.mode == "predcls" else ("pred_scores", "pred_labels")
    future_count = len(future_frame_idx)
    result[score_key] = entry[score_key][first_box:last_box + 1].repeat(future_count)
    result[label_key] = entry[label_key][first_box:last_box + 1].repeat(future_count)
    result["im_idx"] = torch.arange(future_count, device=im_idx.device,
                                    dtype=torch.int32).repeat_interleave(len(pairs))
    result["pair_idx"] = (pairs - first_box).repeat(future_count, 1) + object_count * result["im_idx"][:, None]
    result["boxes"] = torch.full((object_count * future_count, 5), 0.5, device=im_idx.device)
    if persistence_out is not None:
        persistence_out.update(result)
        current = entry["global_output"][mask]
        persistence_out["attention_distribution"] = model.dsgdetr.a_rel_compress(current).repeat(future_count, 1)
        persistence_out["spatial_distribution"] = model.dsgdetr.s_rel_compress(current).sigmoid().repeat(future_count, 1)
        persistence_out["contacting_distribution"] = model.dsgdetr.c_rel_compress(current).sigmoid().repeat(future_count, 1)
    return result
