# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
# 실행: train_and_select.py에서 호출
"""Sol 입력 없이 관측 graph로 학습하는 mask/MTP/rank/memory 비교 조건."""
import torch
from torch.nn import functional as F


def memory_prior(history: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """학습 그룹의 class별 현재 관계 평균만 저장한다."""
    labels = history[:, -1, 26:63].argmax(-1)
    prior = torch.zeros(37, 26)
    for category in range(37):
        selected = current[labels == category]
        prior[category] = selected.mean(0) if len(selected) else current.mean(0)
    return prior


def transform(history: torch.Tensor, method: str, prior: torch.Tensor | None = None, training: bool = False) -> torch.Tensor:
    """추론 때는 관측 입력과 동결 train memory만 사용한다."""
    result = history.clone()
    if method == 'local_masked' and training:
        keep = torch.rand(result.shape[:2], device=result.device) > .25
        keep[:, -1] = True
        result *= keep[..., None]
    if method == 'local_memory':
        assert prior is not None
        labels = result[:, :, 26:63].argmax(-1)
        result[:, :, :26] = .5 * result[:, :, :26] + .5 * prior[labels]
    return result


def ranking_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """한 사례 안의 정답 관계가 비정답 관계보다 높은 순위를 갖도록 학습한다."""
    mask = target[:, :, None].bool() & ~target[:, None, :].bool()
    differences = logits[:, :, None] - logits[:, None, :]
    losses = F.softplus(-differences)
    return (losses * mask).sum() / mask.sum().clamp_min(1)
