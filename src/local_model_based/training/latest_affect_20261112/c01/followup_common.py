# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# How to run: 각 followup 실행기에서 import한다.
"""후속 실행의 원본 보존, 서명, 원자적 저장을 공유한다."""
from pathlib import Path
from typing import Final
import hashlib
import json
import os
import sys
import numpy as np
import torch

ROOT: Final = Path(__file__).resolve().parent
SEED: Final = 20261111


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def freeze(path: Path, value) -> None:
    if path.exists():
        assert json.loads(path.read_text()) == value, path
    else:
        save(path, value)


def metric(prediction: np.ndarray, target: np.ndarray):
    p, y = np.clip(prediction, -1, 1).astype(np.float64).reshape(-1, 8), target.astype(np.float64).reshape(-1, 8)
    covariance = ((p - p.mean(0)) * (y - y.mean(0))).mean(0)
    ccc = 2 * covariance / np.maximum(p.var(0) + y.var(0) + (p.mean(0) - y.mean(0)) ** 2, 1e-15)
    return {'MAE': float(np.abs(p - y).mean()), 'CCC': float(ccc.mean())}


def guard(folder: Path, train: bool) -> None:
    """선택 이전 외부 정답과 네트워크, 기존 산출물 쓰기를 차단한다."""
    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            raise PermissionError('network_forbidden')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if train and (path.name in ('labels_dev.json', 'labels_eval.json') or 'evaluation_only' in path.parts or path.suffix == '.ndjson'):
                raise PermissionError('outer_target_forbidden', str(path))
            writing = args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            if writing and not any(path.is_relative_to(p) for p in (folder, Path('/tmp'), Path('/dev'))):
                raise PermissionError('outside_run_write_forbidden', str(path))
    sys.addaudithook(audit)


def tensor_checkpoint(path: Path, value) -> None:
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)
