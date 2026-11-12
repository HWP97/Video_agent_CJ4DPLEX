"""첫 실험의 읽기 전용 입력을 재사용하는 후속 실험 경로."""
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parents[2] / 'official_local_train_20261111/b1/config_sweep_v3'
SOURCE = Path('/mnt/RG2/paper_experiments_20261109/combinations')
sys.path.insert(0, str(ORIGINAL / 'code'))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
