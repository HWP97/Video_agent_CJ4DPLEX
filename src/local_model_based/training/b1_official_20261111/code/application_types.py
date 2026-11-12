# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""Typed boundaries for the relation-only application adapter."""
from typing import TypedDict
import torch


class Prediction(TypedDict):
    boxes: torch.Tensor
    labels: torch.Tensor
    pred_labels: torch.Tensor
    scores: torch.Tensor
    pred_scores: torch.Tensor
    pair_idx: torch.Tensor
    im_idx: torch.Tensor
    attention_distribution: torch.Tensor
    spatial_distribution: torch.Tensor
    contacting_distribution: torch.Tensor


class Bundle(TypedDict, total=False):
    state_dict: dict[str, torch.Tensor]
    residual: bool
    anchor: bool
    method: str
    mode: str
    seed: int
    schema_version: int
    epochs: int
    training_groups: int
    protocol_sha256: str
    encoder_checkpoint_sha256: str
    inference_ablation: str
    time_origin: str
    time_scale: float


class Supervision(TypedDict):
    predicted_history: torch.Tensor
    gt_history: torch.Tensor
    current: torch.Tensor
    base_logits: torch.Tensor
    future: torch.Tensor
    delta: torch.Tensor
    coverage: dict[str, int | str | list[int]]


class AdapterInputError(ValueError):
    """Frozen model and observed graph do not satisfy the inference contract."""


class TargetAlignmentError(ValueError):
    """An eligible annotated input cannot supply the declared training target."""
