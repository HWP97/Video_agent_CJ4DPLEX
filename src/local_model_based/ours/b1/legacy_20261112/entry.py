# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch"]
# ///
# 실행: python main.py b1 ours predict --sample predcls:00607.mp4_0.5
"""선정 pretrained ImparTail 유지. 추가 head는 채택하지 않았다."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'baseline/b1'))
from runner import main

if __name__ == '__main__':
    main('ours')
