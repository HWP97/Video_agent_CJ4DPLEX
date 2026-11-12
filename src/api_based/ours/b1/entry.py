# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: python main.py b1 ours evaluate --limit 1
"""기존 macro_direct를 전 조건에 고정 적용하는 사후 선택 실행."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ours_cli import main

if __name__ == '__main__':
    main()
