# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# 실행: 다른 모듈이 경로 상수만 import
"""기존 공식 train 500영상 부분집합 실험의 독립 출력 경로."""
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
