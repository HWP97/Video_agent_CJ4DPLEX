# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scipy", "tokenizers"]
# ///
# How to run: train.py, infer.py, score.py에서 import한다.
"""읽기 전용 기존 모델과 데이터 계약을 공유한다."""
from pathlib import Path
from typing import Final
import json
import os
import sys
import hashlib
import numpy as np
import torch

FOLDER: Final = Path(__file__).resolve().parent
ROOT: Final = FOLDER.parents[1]
OLD: Final = FOLDER.parent / 'followup_reasoned_combinations'
ORIGINAL: Final = ROOT.parent / 'official_local_train_20261111/b3'
SEEDS: Final = (20261111, 20261110)
EPOCHS: Final = (1, 3, 5, 10, 20, 25)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OLD))
from followup_common import sha, save, freeze, guard, tensor_checkpoint
from models import Model, CONFIGS, features, objectives
from engine import predict, batch

Data = dict[str, torch.Tensor]


def setup(training: bool = True) -> None:
    """기존 환경의 결정론 연산과 파일/네트워크 경계를 고정한다."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    guard(FOLDER, training)


def ids_sha(ids: np.ndarray) -> str:
    return hashlib.sha256(ids.tobytes()).hexdigest()


def load_data(fold: int, split: int | None) -> tuple[Data, np.ndarray, np.ndarray, np.ndarray]:
    """학습 부분에서만 통계를 fit하며 원본 그룹 전체를 가린다."""
    folder = FOLDER.parent / 'data' / f'fold{fold}'
    receipt = json.loads((folder / 'receipt.json').read_text())
    assert sha(folder / 'training.npz') == receipt['train_array_sha256']
    registration = json.loads((FOLDER / 'registration.json').read_text())
    with np.load(folder / 'training.npz') as values:
        data = {k: torch.as_tensor(values[k], dtype=torch.float32, device='cuda') for k in values.files if k not in ('groups', 'inner_train', 'inner_validation')}
        groups = values['groups'].copy()
        if split is None:
            train = np.arange(len(groups))
            validation = train[:min(2048, len(train))]
        else:
            group = registration['folds'][str(fold)]['validation_groups'][split]
            train, validation = np.flatnonzero(groups != group), np.flatnonzero(groups == group)
            assert not set(groups[train]) & set(groups[validation])
    x = features(data)[train]
    data['center'], data['spread'] = x.mean(0), x.std(0, unbiased=False).clamp_min(.05)
    return data, train, validation, groups


def errors(model: Model, data: Data, validation: np.ndarray) -> tuple[np.ndarray, float, float]:
    """학습 영역 내부의 실제 metric을 계산한다."""
    outputs, _ = predict(model, data, validation)
    target = data['target'][validation].cpu().numpy()
    scale = data['scale'][validation].cpu().numpy()
    distance = np.linalg.norm(outputs[:, -1] - target, axis=-1) * scale[:, None]
    return outputs, float(distance.mean()), float(distance[:, -1].mean())
