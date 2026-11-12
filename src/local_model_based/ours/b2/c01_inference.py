# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: latest_inference.py에서 호출
"""C01의 동결 PCA32·GRU8·horizon head를 실제 계산한다."""
from pathlib import Path
import sys
from typing import Final

import numpy as np
from numpy.typing import NDArray
import torch

ROOT: Final = Path(__file__).resolve().parents[2]
DATA: Final = ROOT / "data/b2/latest"
sys.path.insert(0, str(ROOT / "training/latest_affect_20261112"))
from c01.models import Expert


def predict_c01(raw: NDArray, numeric: NDArray, hold: NDArray, times: NDArray) -> NDArray:
    """원 실험과 같은 chunk·정규화·float32 연산으로 직접 미래값을 출력한다."""
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with np.load(DATA / "model/c01/transform.npz", allow_pickle=False) as t:
        parts = [((raw[i:i+128] - t["mean"]) / t["scale"]) @ t["basis"] / np.sqrt(32) for i in range(0, len(raw), 128)]
        x = np.concatenate((np.concatenate(parts), ((times-times[:, -1:])/8)[:, :, None]), -1).astype(np.float32)
        n = ((numeric - t["nm"]) / t["ns"]).astype(np.float32)
    model = Expert().eval()
    saved = torch.load(DATA / "model/c01/checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state"], strict=True)
    x_tensor, numeric_tensor = torch.from_numpy(x), torch.from_numpy(n)
    with torch.inference_mode():
        expert = torch.cat([model.predict(x_tensor[ids], numeric_tensor[ids]) for ids in np.array_split(np.arange(len(raw)), max(1, (len(raw)+255)//256))])
        # fixed_gate=1, a=1, b=0인 원 C01의 float32 연산 순서만 보존한다.
        held = torch.from_numpy(np.repeat(hold[:, None], 4, axis=1).astype(np.float32))
        return (held + (expert-held)).clamp(-1, 1).numpy()
