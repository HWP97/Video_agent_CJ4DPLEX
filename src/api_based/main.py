# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# 실행: python3 -B main.py b2 final evaluate
"""복사된 branch만 실행하는 오프라인 진입점."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Final
from bundle_paths import RESULTS

ROOT: Final = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="복사본 실행: baseline / ours, 신규 API 없음 (final 호환 별칭)")
    parser.add_argument("branch", nargs="?", choices=("list", "b1", "b2", "b3"), default="list")
    parser.add_argument("stage", nargs="?", choices=("baseline", "ours", "final"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    parser.add_argument("--python", dest="python_path", help="기존 호환 환경 Python 경로")
    args = parser.parse_args()
    if args.branch == "list":
        for branch in ("b1", "b2", "b3"):
            for directory in ("baseline", "ours"):
                metadata = ROOT / directory / branch / "branch.json"
                if metadata.exists():
                    print(metadata.relative_to(ROOT))
                    print(metadata.read_text(encoding="utf-8"))
        return 0
    if args.stage is None:
        parser.error("baseline 또는 ours를 선택하세요")
    directory = {"baseline": "baseline", "ours": "ours", "final": "ours"}[args.stage]
    entry = ROOT / directory / args.branch / "entry.py"
    metadata = entry.with_name("branch.json")
    if not entry.is_file() or not metadata.is_file():
        parser.error(f"branch 준비 파일이 없습니다: {entry}")
    config = json.loads(metadata.read_text(encoding="utf-8"))
    executable = Path(args.python_path or config.get("python") or sys.executable)
    if not executable.is_file():
        parser.error(f"호환 Python이 없습니다. --python으로 지정하세요: {executable}")
    arguments = args.arguments or ["status"]
    environment = {key: value for key, value in os.environ.items()
                   if not any(marker in key.upper() for marker in ("API_KEY", "ACCESS_TOKEN", "SECRET", "AUTH_TOKEN", "HF_TOKEN"))
                   and key not in ("PYTHONPATH", "PYTHONSTARTUP")}
    scratch = RESULTS / "_runtime"
    scratch.mkdir(parents=True, exist_ok=True)
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TMPDIR": str(scratch), "XDG_CACHE_HOME": str(scratch / "cache"),
        "HF_HOME": str(scratch / "hf"), "MPLCONFIGDIR": str(scratch / "matplotlib"),
        "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2",
    })
    command = [str(executable), "-B", str(ROOT / "runtime_guard.py"), str(entry), *arguments]
    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
