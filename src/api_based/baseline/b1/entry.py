# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: python main.py b1 baseline status
"""저장된 Sol 응답 전용 b1 진입점."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import main

if __name__ == '__main__':
    main('baseline')
