# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
# 실행: run.py에서 import; test_selector.py로 검증.
"""GT 없이 클래스/방향/술어 중복을 분산하는 WC 후보 선택."""
from dataclasses import replace
import numpy as np
from adapters import Entry

def select(entry: Entry, diversify: bool) -> Entry:
    """행과 물체는 유지하고 중복 행은 같은 family의 다음 미사용 술어를 선택한다."""
    if not len(entry.pairs):
        return entry
    if not np.isfinite(entry.relations).all():
        raise ArithmeticError("nonfinite candidate scores")
    scores = entry.relations.copy()
    confidence = scores.max(1) * entry.objects[entry.pairs].prod(1)
    initial = np.argsort(-confidence, kind='stable')
    used: set[tuple[int, int, int]] = set()
    unique, duplicate = [], []
    for row in initial:
        classes = tuple(int(v) for v in entry.classes[entry.pairs[row]])
        candidates = np.argsort(-scores[row], kind='stable')
        active = candidates[scores[row, candidates] > 0]
        winner = int(candidates[0])
        if diversify:
            available = [int(p) for p in active if (*classes, int(p)) not in used]
            if available:
                winner = available[0]
                # 다른 family의 0점 후보를 만들지 않고 원래 점수로 한 후보만 남긴다.
                value = scores[row, winner]
                scores[row] = 0
                scores[row, winner] = value
        key = (*classes, winner)
        (duplicate if key in used else unique).append(int(row))
        used.add(key)
    priority = scores.max(1) * entry.objects[entry.pairs].prod(1)
    order = np.array(sorted(unique, key=lambda r: -priority[r]) + sorted(duplicate, key=lambda r: -priority[r]))
    return replace(entry, pairs=entry.pairs[order], relations=scores[order])
