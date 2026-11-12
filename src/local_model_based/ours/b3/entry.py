# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scipy", "pydantic", "tokenizers"]
# ///
# How to run: main.py b3 ours status|replay|predict|evaluate|verify
"""최신 동일3fold 보행 모델 진입점. 이전 구현은 legacy_20261112에 보존한다."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baseline/b3"))
from latest_entry import main

if __name__ == "__main__":
    main("ours", Path(__file__))

