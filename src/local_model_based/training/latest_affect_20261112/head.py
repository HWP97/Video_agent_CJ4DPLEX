# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# How to run: train.py / infer.py에서 import
"""관측 전용 변환과 작은 ridge 잔차 head의 공통 수학."""
from pathlib import Path
import numpy as np
import torch
from prepare import OUT, SOURCE


def features(train: np.ndarray, folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """모든 fit 통계는 전달된 학습 ID만 사용한다."""
    with np.load(SOURCE/'inputs.npz') as saved:
        raw, numeric, recognition = saved['x'], saved['numeric'], saved['recognition']
    visual = np.concatenate((raw[:,-1],raw.mean(1)),1).astype(np.float64)
    path = folder/'transform.npz'
    if not path.exists():
        mean,scale = visual[train].mean(0), np.maximum(visual[train].std(0),1e-4)
        torch.manual_seed(20261111)
        standardized = torch.from_numpy(((visual[train]-mean)/scale).astype(np.float32))
        _,_,basis = torch.pca_lowrank(standardized,q=20,center=False,niter=3)
        basis_np = basis[:,:16].numpy().astype(np.float64)
        projected = ((visual-mean)/scale)@basis_np
        joined = np.concatenate((projected,numeric),1)
        fm,fs = joined[train].mean(0),np.maximum(joined[train].std(0),1e-4)
        folder.mkdir(parents=True,exist_ok=True)
        np.savez(path,mean=mean,scale=scale,basis=basis_np,fm=fm,fs=fs,fit_ids=train)
    with np.load(path) as t:
        projected = ((visual-t['mean'])/t['scale'])@t['basis']
        joined = np.concatenate((projected,numeric),1)
        x = (joined-t['fm'])/t['fs']
    return np.column_stack((x,np.ones(len(x)))), recognition[:,-1].astype(np.float64)


def ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """정규화된 사례/그룹 손실과 절편 제외 ridge를 정확히 푼다."""
    penalty = np.eye(x.shape[1])*.01
    penalty[-1,-1] = 0
    normalized = weights/weights.sum()
    return np.linalg.solve(x.T@(normalized[:,None]*x)+penalty,x.T@(normalized[:,None]*y))


def current_metric(p: np.ndarray, y: np.ndarray) -> tuple[float,float]:
    """기존 미래 CCC와 같은 모집단 분산 정의, 현재 VA의 두 채널 평균."""
    prediction = p.clip(-1,1).astype(np.float64)
    target = y.astype(np.float64)
    covariance = ((prediction-prediction.mean(0))*(target-target.mean(0))).mean(0)
    ccc = 2*covariance/np.maximum(prediction.var(0)+target.var(0)+(prediction.mean(0)-target.mean(0))**2,1e-15)
    return float(np.abs(prediction-target).mean()),float(ccc.mean())
