"""게시 실패 후 실제 격리 프로세스가 계속 실행되는지 확인한다."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

import publish_parallel as publisher


@pytest.mark.parametrize('failure', ('temporary_cleanup', 'rollback'))
def test_process_resumes_when_cleanup_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          failure: str) -> None:
    name = 'attention_moif_default'
    root = tmp_path / 'adapters/reduced_papers'
    source = root / 'parallel_fold1_gpu1'
    destination = root / 'fold1'
    for folder in (source / 'fit_dev', destination / 'fit_dev', root / 'evidence'):
        folder.mkdir(parents=True, exist_ok=True)
    runner = root / 'parallel_attention.py'
    runner.write_text('fixture')
    signature = {'sources': publisher.sources(), 'kind': 'attention', 'seed': 20261109,
                 'probe': False, 'config': asdict(publisher.model_module().CONFIGS[4]),
                 'train': {'fixture': 'train'}, 'dev': {'fixture': 'dev'}}
    (destination / 're_default.receipt.json').write_text(json.dumps({'signature': signature}))
    torch.save({'signature': signature}, source / f'{name}.pt')
    np.save(source / 'fit_dev' / f'{name}.npy', np.zeros((2219, 12, 2)))
    receipt = {'probe_only': False, 'signature': signature,
               'checkpoint_sha256': publisher.sha(source / f'{name}.pt'),
               'prediction_sha256': publisher.sha(source / 'fit_dev' / f'{name}.npy')}
    (source / f'{name}.receipt.json').write_text(json.dumps(receipt))
    (source / f'{name}.producer.json').write_text(json.dumps({
        'runner_sha256': publisher.sha(runner), 'physical_gpu': '1', 'probe_only': False}))
    current = 'attention_re_partitions4' if failure == 'temporary_cleanup' else 're_default'
    (root / 'evidence/train_fold1.log').write_text(json.dumps({'method': current}) + '\n')
    monkeypatch.setattr(publisher, 'ROOT', tmp_path)
    monkeypatch.setattr(publisher, 'install_guard', lambda *_: None)
    monkeypatch.setattr(sys, 'argv', ['publish_parallel.py', name, '--publish'])
    original_unlink, original_replace = Path.unlink, Path.replace

    def failing_unlink(path: Path, missing_ok: bool = False) -> None:
        if (failure == 'temporary_cleanup' and '.parallel-' in path.name) or (
                failure == 'rollback' and path == destination / f'{name}.pt'):
            raise PermissionError(path)
        original_unlink(path, missing_ok=missing_ok)

    def failing_replace(path: Path, target: Path) -> Path:
        if failure == 'rollback' and target.suffix == '.npy':
            raise PermissionError(target)
        return original_replace(path, target)

    with subprocess.Popen(['sleep', '60']) as process:
        monkeypatch.setattr(publisher, 'PID', process.pid)

        def state() -> str:
            return Path(f'/proc/{process.pid}/stat').read_text().split()[2]

        monkeypatch.setattr(publisher, 'process_state', state)
        monkeypatch.setattr(Path, 'unlink', failing_unlink)
        monkeypatch.setattr(Path, 'replace', failing_replace)
        try:
            with pytest.raises(PermissionError):
                publisher.main()
            deadline = time.monotonic() + 2
            while state() in ('T', 't') and time.monotonic() < deadline:
                os.sched_yield()
            assert state() not in ('T', 't'), '정리 실패 뒤 격리 프로세스가 정지 상태로 남음'
        finally:
            os.kill(process.pid, signal.SIGCONT)
            process.terminate()

