# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest"]
# ///
# 실행: 기존 pytest 환경에서 python -B -m pytest tests/test_runtime.py
"""실제 복사 런처의 외부 요청/원본 접근과 환경 분리 계약."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def bundle(tmp_path: Path, program: str) -> Path:
    destination = tmp_path / "family"
    entry = destination / "baseline" / "b1"
    entry.mkdir(parents=True)
    for name in ("main.py", "runtime_guard.py", "bundle_paths.py"):
        shutil.copyfile(ROOT / name, destination / name)
    (entry / "entry.py").write_text(program, encoding="utf-8")
    (entry / "branch.json").write_text(json.dumps({"python": sys.executable}), encoding="utf-8")
    return destination


def execute(folder: Path, *arguments: str) -> subprocess.CompletedProcess:
    environment = dict(os.environ, OPENAI_API_KEY="test-only-placeholder",
                       GEMINI_API_KEY="test-only-placeholder", PYTHONPATH="/invalid/old/code")
    return subprocess.run([sys.executable, "-B", str(folder / "main.py"), "b1", "baseline", *arguments],
                          env=environment, text=True, capture_output=True, check=False)


def test_real_entry_reads_own_data_without_keys(tmp_path: Path) -> None:
    folder = bundle(tmp_path, """import json, os
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'payload.json'
print(json.dumps({'value':json.loads(p.read_text()), 'key_present':'OPENAI_API_KEY' in os.environ,
                  'other_key':'GEMINI_API_KEY' in os.environ, 'offline':os.environ['HF_HUB_OFFLINE']}))
""")
    (folder / "payload.json").write_text('{"id":"actual-copy"}')
    result = execute(folder, "replay")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"value": {"id": "actual-copy"}, "key_present": False,
                                        "other_key": False, "offline": "1"}


@pytest.mark.parametrize("action", ("read", "write", "rename", "network", "process"))
def test_live_boundary_stops_external_actions(tmp_path: Path, action: str) -> None:
    original = tmp_path / "original.json"
    original.write_text('{"keep":true}')
    actions = {
        "read": "Path(__file__).resolve().parents[3].joinpath('original.json').read_text()",
        "write": "Path(__file__).resolve().parents[3].joinpath('original.json').write_text('changed')",
        "rename": "Path(__file__).resolve().parents[3].joinpath('original.json').rename(Path(__file__).parent/'stolen.json')",
        "network": "socket.socket().connect(('127.0.0.1',9))",
        "process": "subprocess.run([sys.executable,'-c','print(1)'],check=True)",
    }
    folder = bundle(tmp_path, "from pathlib import Path\nimport socket, subprocess, sys\n" + actions[action])
    result = execute(folder, "verify")
    assert result.returncode != 0
    assert "offline bundle boundary" in result.stderr
    assert original.read_text() == '{"keep":true}'


def test_missing_cache_has_no_recovery(tmp_path: Path) -> None:
    folder = bundle(tmp_path, """from pathlib import Path
Path(__file__).with_name('missing_response.json').read_text()
""")
    result = execute(folder, "replay")
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
