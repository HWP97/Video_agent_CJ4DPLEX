# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "pytest"]
# ///
# 실행: 기존 AG Python -m pytest code/test_extensions.py
"""새 loss의 학습 방향과 memory/mask 입력 경계를 검증한다."""
import torch
from local_extensions import memory_prior, transform, ranking_loss


def test_ranking_gradient_promotes_true_relation() -> None:
    logits = torch.zeros(1, 3, requires_grad=True)
    ranking_loss(logits, torch.tensor([[1.,0.,0.]])).backward()
    assert logits.grad[0,0] < 0
    assert torch.all(logits.grad[0,1:] > 0)


def test_training_mask_preserves_latest_observation() -> None:
    torch.manual_seed(7)
    history = torch.ones(100,4,71)
    output = transform(history,'local_masked',training=True)
    assert torch.equal(output[:,-1],history[:,-1])
    assert (output[:,:-1] == 0).any()
    assert torch.equal(transform(history,'local_masked'),history)


def test_memory_is_class_conditioned_and_handles_unseen_class() -> None:
    history = torch.zeros(2,4,71)
    history[0,:,28] = 1
    history[1,:,29] = 1
    targets = torch.stack((torch.zeros(26),torch.ones(26)))
    prior = memory_prior(history,targets)
    assert torch.equal(prior[2],torch.zeros(26))
    assert torch.equal(prior[3],torch.ones(26))
    assert torch.equal(prior[4],torch.full((26,),.5))
    output = transform(history,'local_memory',prior)
    assert torch.equal(output[1,:,:26],torch.full((4,26),.5))
    assert torch.equal(output[:,:,26:],history[:,:,26:])
