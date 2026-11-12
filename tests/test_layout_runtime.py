"""실제 런처로 분리된 결과 쓰기와 원본·다른 계열 접근 경계를 확인한다."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def prepare(tmp_path, program):
    family = tmp_path / "submission/src/local_model_based"
    entry = family / "baseline/b1"
    entry.mkdir(parents=True)
    for name in ("main.py", "runtime_guard.py", "bundle_paths.py"):
        shutil.copyfile(PROJECT / "src/local_model_based" / name, family / name)
    (entry / "entry.py").write_text(program)
    (entry / "branch.json").write_text(json.dumps({"python": sys.executable}))
    return family


def run(family):
    return subprocess.run([sys.executable, "-B", str(family / "main.py"), "b1", "baseline", "predict"],
                          text=True, capture_output=True)


def test_results_outside_src_are_writable(tmp_path):
    family = prepare(tmp_path, "from bundle_paths import RESULTS\n(RESULTS/'actual.json').write_text('{\"ok\":true}')")
    result = run(family)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "submission/results/local_model_based/actual.json").read_text()) == {"ok": True}
    assert not (family / "results").exists()


@pytest.mark.parametrize("stage", ["ours", "final"])
def test_ours_and_legacy_final_route_to_ours(tmp_path, stage):
    family = prepare(tmp_path, "raise RuntimeError('baseline must not run')")
    ours = family / 'ours/b1'
    ours.mkdir(parents=True)
    shutil.copyfile(family / 'baseline/b1/branch.json', ours / 'branch.json')
    (ours / 'entry.py').write_text("from bundle_paths import RESULTS\n(RESULTS/'ours-ran').write_text('actual ours')")
    result = subprocess.run([sys.executable, '-B', str(family / 'main.py'), 'b1', stage, 'replay'],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'submission/results/local_model_based/ours-ran').read_text() == 'actual ours'


@pytest.mark.parametrize("target", ["original.json", "submission/src/api_based/private.json", "submission/results/api_based/private.json"])
def test_original_and_other_family_are_unreadable(tmp_path, target):
    source = tmp_path / target
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("preserve")
    family = prepare(tmp_path, f"from pathlib import Path\nPath({str(source)!r}).read_text()")
    result = run(family)
    assert result.returncode != 0
    assert "read_original_workspace" in result.stderr
    assert source.read_text() == "preserve"
