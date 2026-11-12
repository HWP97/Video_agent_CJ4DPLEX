# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "pytest"]
# ///
# 실행: 기존 AG Python -m pytest tests/test_log_provenance.py
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from log_objective import teacher_kl,restored_scores
from provenance import freeze


def test_log_teacher_is_stationary_at_baseline_scores() -> None:
    base=torch.linspace(.01,.99,26)[None]
    logs=base.log().detach().requires_grad_()
    teacher_kl(logs,base).backward()
    assert float(logs.grad.abs().max())<1e-7


def test_zero_correction_preserves_scores_and_positive_scale_is_not_clipped() -> None:
    base=torch.tensor([[.01,.5,.9]])
    assert torch.equal(restored_scores(base.log(),base),base)
    shifted=restored_scores(base.log()+.75,base)
    assert shifted[0,-1]>1
    assert torch.allclose(shifted,base*torch.exp(torch.tensor(.75)))


def test_zero_correction_keeps_raw_zero_despite_log_floor() -> None:
    raw=torch.tensor([[0.,1e-8,.5]])
    model_base=raw.clamp(1e-5,1-1e-5)
    assert torch.equal(restored_scores(model_base.log(),model_base,raw),raw)


def test_changed_freeze_fails_without_overwriting_original(tmp_path: Path) -> None:
    path=tmp_path/'freeze.json'
    original=freeze(path,{'model':'a','data':'x'})
    assert freeze(path,{'model':'a','data':'x'})==original
    with pytest.raises(AssertionError,match='Frozen provenance mismatch'):
        freeze(path,{'model':'b','data':'x'})
    assert freeze(path,{'model':'a','data':'x'})==original
