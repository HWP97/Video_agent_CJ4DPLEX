"""Inference-only compatibility for the pinned SceneSayer source and PyTorch 2.5."""
import sys
import types

import torch
from torchvision import ops


def inclusive_nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    # The original CUDA kernel uses inclusive pixel corners (+1 area).
    adjusted = boxes.clone()
    adjusted[:, 2:] += 1
    return ops.nms(adjusted, scores, threshold)


def install_roi_compat() -> None:
    """Replace obsolete THC extension imports, preserving legacy ROI coordinates."""
    module = types.ModuleType("fasterRCNN.lib.model.roi_layers")
    module.nms = inclusive_nms
    module.ROIAlign = ops.RoIAlign
    module.roi_align = ops.roi_align
    module.ROIPool = ops.RoIPool
    module.roi_pool = ops.roi_pool
    sys.modules[module.__name__] = module

