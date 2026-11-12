# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
# 실행: runtime_guard.py ours/b2/legacy_entry.py evaluate
"""교체 전 localpool Ridge 실행 경로를 보존한다."""
from inference import main

if __name__ == "__main__":
    main()
