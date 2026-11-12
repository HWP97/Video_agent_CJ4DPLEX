# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
# 실행: runtime_guard.py baseline/b2/legacy_entry.py evaluate
"""교체 전 hold 평가와 실제 JPEG 인식 명령을 보존한다."""
from runtime import main

if __name__ == "__main__":
    main("baseline")
