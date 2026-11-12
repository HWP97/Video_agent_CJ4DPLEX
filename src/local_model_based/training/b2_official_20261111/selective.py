# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "scikit-learn"]
# ///
# How to run: python -B run.py
"""Departure classification and original-group envelope CRC."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

@dataclass(frozen=True)
class FitResult:
    probabilities: NDArray[np.float64]
    threshold: float
    c: float
    development: tuple[dict, ...]
    coefficients: dict


def fit_departure(data) -> FitResult:
    """Three regularization variants; all model selection is grouped OOF."""
    dev = data.split=='development'
    x = data.features[dev].reshape(-1,7)
    magnitude = np.linalg.norm(data.truth-data.current_gt[:,None,:],axis=-1)
    distances = magnitude[dev].ravel()
    groups = np.repeat(data.groups[dev],4)
    evaluations = []
    for c in (.1,1.,10.):
        oof = np.zeros(len(x))
        truth = np.zeros(len(x),dtype=bool)
        for train, validation in GroupKFold(3).split(x,groups=groups):
            threshold = np.quantile(distances[train],.75)
            y = distances[train] > threshold
            model = make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000,random_state=20261109))
            model.fit(x[train],y)
            oof[validation] = model.predict_proba(x[validation])[:,1]
            truth[validation] = distances[validation] > threshold
        evaluations.append({'C':c,'OOF_AUPRC':float(average_precision_score(truth,oof)),
            'OOF_AUROC':float(roc_auc_score(truth,oof)),'OOF_prior':float(truth.mean())})
    selected = max(evaluations,key=lambda r:(r['OOF_AUPRC'],-r['C']))
    threshold = float(np.quantile(distances,.75))
    model = make_pipeline(StandardScaler(),LogisticRegression(C=selected['C'],max_iter=1000,random_state=20261109))
    model.fit(x,distances>threshold)
    p = model.predict_proba(data.features.reshape(-1,7))[:,1].reshape(-1,4)
    scaler, classifier = model.steps[0][1],model.steps[1][1]
    coefficients = {'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
        'coef':classifier.coef_.tolist(),'intercept':classifier.intercept_.tolist()}
    return FitResult(p,threshold,selected['C'],tuple(evaluations),coefficients)


def crc_threshold(confidence: NDArray[np.float64], errors: NDArray[np.bool_], groups: NDArray[np.str_], alpha: float) -> tuple[float, dict]:
    """Eq3: suffix maximum per group before group-equal averaging."""
    grid = np.r_[np.linspace(.5,1.,51),1.01]
    losses=[]
    for group in np.unique(groups):
        member = groups==group
        losses.append([float(errors[member & (confidence>=v)].mean()) if np.any(member & (confidence>=v)) else 0. for v in grid])
    loss = np.array(losses)
    envelope = np.maximum.accumulate(loss[:,::-1],axis=1)[:,::-1]
    n=len(loss)
    bound=(envelope.sum(axis=0)+1)/(n+1)
    eligible=np.flatnonzero(bound<=alpha)
    chosen=float(grid[eligible[0]]) if len(eligible) else float(grid[-1])
    return chosen,{'groups':n,'grid':grid.tolist(),'raw_group_losses':loss.tolist(),
        'envelope_group_losses':envelope.tolist(),'corrected_bound':bound.tolist()}


def departure_metrics(truth: NDArray[np.bool_], probability: NDArray[np.float64], committed: NDArray[np.bool_], groups: NDArray[np.str_], horizons: NDArray[np.float64]) -> dict:
    """Micro and group-equal errors plus alarm recall include abstentions."""
    decision=probability>=.5
    wrong=decision!=truth
    tp=int(np.sum(committed & decision & truth))
    fp=int(np.sum(committed & decision & ~truth))
    alarms=int(np.sum(committed & decision))
    total_positive=int(truth.sum())
    precision=tp/alarms if alarms else 0.
    recall=tp/total_positive if total_positive else 0.
    lt_recall=float(np.sum((committed & decision & truth)*np.minimum(horizons/3,1)))/total_positive if total_positive else 0.
    group_loss=[]
    conditional=[]
    for group in np.unique(groups):
        mask=committed & (groups==group)
        loss=float(wrong[mask].mean()) if mask.any() else 0.
        group_loss.append(loss)
        if mask.any(): conditional.append(loss)
    return {'n':len(truth),'committed':int(committed.sum()),'coverage':float(committed.mean()),
        'error':float(wrong[committed].mean()) if committed.any() else None,
        'group_macro_risk':float(np.mean(group_loss)),
        'conditional_group_risk':float(np.mean(conditional)) if conditional else None,
        'departure_prevalence':float(truth.mean()),'AUPRC':float(average_precision_score(truth,probability)),
        'AUROC':float(roc_auc_score(truth,probability)) if len(np.unique(truth))==2 else None,
        'alarm_precision':precision,'alarm_recall':recall,'TP':tp,'FP':fp,
        'shift_F1':2*precision*recall/(precision+recall) if precision+recall else 0.,
        'seconds_LT_F1':2*precision*lt_recall/(precision+lt_recall) if precision+lt_recall else 0.}


def va_metrics(truth: NDArray[np.float64], prediction: NDArray[np.float64]) -> dict:
    """Mean over eight separately pooled axis/horizon CCC values."""
    y=truth.reshape(len(truth),-1)
    p=prediction.reshape(len(prediction),-1)
    denominator=y.var(axis=0)+p.var(axis=0)+(y.mean(axis=0)-p.mean(axis=0))**2
    ccc=2*((y-y.mean(axis=0))*(p-p.mean(axis=0))).mean(axis=0)/np.maximum(denominator,1e-15)
    return {'n':len(y),'MAE':float(np.abs(y-p).mean()),'CCC':float(ccc.mean()),'CCC_dimensions':ccc.tolist()}
