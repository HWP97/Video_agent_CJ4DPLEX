# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# 실행: python3 main.py --help
"""API·로컬 모델 재현과 영상 데모 렌더링의 제출용 진입점."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Final

ROOT: Final = Path(__file__).resolve().parent
TASKS: Final = {"relations": "b1", "affect": "b2", "trajectories": "b3"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", help="다른 장비에서 사용할 호환 Python 경로")
    parser.add_argument("command", choices=("api", "local", "demo"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = list(args.arguments)
    if arguments:
        arguments[0] = TASKS.get(arguments[0], arguments[0])
    if args.command in ("api", "local"):
        environment_option = ["--python", args.python] if args.python else []
        family = {"api": "api_based", "local": "local_model_based"}[args.command]
        return subprocess.run([sys.executable, "-B", str(ROOT / "src" / family / "main.py"),
            *environment_option, *arguments], cwd=ROOT, check=False).returncode
    if not arguments or arguments[0] not in ("b1", "b2", "b3"):
        parser.error("demo 다음에 relations, affect 또는 trajectories 를 지정하세요")
    python = args.python or json.loads((ROOT / "src/api_based/baseline/b1/branch.json").read_text())["python"]
    script_name = {"b1": "b1_demo", "b2": "b2", "b3": "b3"}[arguments[0]]
    script = ROOT / "tools/api_demo" / (script_name + ".py")
    return subprocess.run([python, "-B", str(script), *arguments[1:]], cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
