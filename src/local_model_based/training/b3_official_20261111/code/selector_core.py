"""Observed-history-only motion selection; the legacy forecast stays unchanged."""
from dataclasses import dataclass
from typing import Final, Literal, TypedDict
import math
import numpy as np
from trajectory_baselines import forecast

Method = Literal['hold', 'constant_velocity', 'kalman']
METHODS: Final = ('hold', 'constant_velocity', 'kalman')

class HistoryPoint(TypedDict):
    time_s: float
    x: float
    y: float

@dataclass(frozen=True, slots=True)
class SelectorResult:
    points: list[HistoryPoint]
    selected: str
    residuals: dict[str, float | None]
    virtual_horizons_s: tuple[float, ...]
    origin_cutoffs_s: tuple[float, ...]
    fallback: str | None


def select_forecast(history: list[HistoryPoint], future_times_s: list[float], virtual_steps: int = 1) -> SelectorResult:
    """All methods fit identical prefixes; only subsequent observed points score them."""
    if virtual_steps not in (1, 2, 3):
        raise ValueError('virtual_steps must be one of frozen candidates 1,2,3')
    if len(history) < 3:
        raise ValueError('legacy baseline requires at least three observations')
    origins = list(range(max(3, len(history)-virtual_steps-2), len(history)-virtual_steps+1))
    errors = {method: [] for method in METHODS}
    horizons, cutoffs = [], []
    for origin in origins:
        prefix, truth = history[:origin], history[origin:origin+virtual_steps]
        times = [p['time_s'] for p in truth]
        cutoffs.append(prefix[-1]['time_s'])
        horizons.extend(t-prefix[-1]['time_s'] for t in times)
        for method in METHODS:
            prediction = forecast(prefix, times, method)
            errors[method].extend(math.hypot(a['x']-b['x'], a['y']-b['y']) for a,b in zip(prediction,truth))
    residuals = {m: float(np.mean(e)) if e else None for m,e in errors.items()}
    selected = 'hold'
    if origins:
        best = float('inf')
        for method in METHODS:
            value = residuals[method]
            if value is not None and value < best-1e-12:
                selected, best = method, value
    return SelectorResult(forecast(history, future_times_s, selected), selected, residuals,
                          tuple(horizons), tuple(cutoffs), None if origins else 'no_three_point_prefix_plus_virtual_future')


def damped_forecast(history: list[HistoryPoint], future_times_s: list[float], decay: float) -> list[HistoryPoint]:
    """OLS velocity decays continuously per second; endpoints0/1 exactly hold/CV."""
    if not 0 <= decay <= 1:
        raise ValueError('decay outside frozen range [0,1]')
    cv = forecast(history, future_times_s, 'constant_velocity')
    if decay == 1:
        return cv
    if decay == 0:
        return forecast(history, future_times_s, 'hold')
    start = history[-1]
    rate = -math.log(decay)
    result = []
    for p in cv:
        dt = p['time_s']-start['time_s']
        factor = -math.expm1(-rate*dt)/(rate*dt)
        result.append({'time_s':p['time_s'],'x':start['x']+(p['x']-start['x'])*factor,
                       'y':start['y']+(p['y']-start['y'])*factor})
    return result
