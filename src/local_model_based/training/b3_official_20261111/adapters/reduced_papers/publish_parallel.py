"""본인 GPU0 실행기를 잠깐 정지하고 검증된 별도 fit 결과만 게시한다."""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from adapter_io import install_guard, sha
from adapter_reduced import model_module, save, sources

PID = 2574995
START_TICKS = '505722253'


def process_state() -> str:
    proc = Path('/proc') / str(PID)
    assert proc.stat().st_uid == os.getuid()
    fields = (proc / 'stat').read_text().split()
    assert fields[21] == START_TICKS
    args = (proc / 'cmdline').read_bytes().split(b'\0')
    assert str(ROOT / 'code/adapter_reduced.py').encode() in args
    assert args[-3:-1] == [b'train', b'1']
    return fields[2]


def main() -> None:
    name = sys.argv[1]
    publish = sys.argv[2:] == ['--publish']
    assert publish or not sys.argv[2:]
    assert name in ('attention_moif_default', 'attention_moif_first_only',
                    'attention_moif_no_direction', 'attention_moif_eta_zero')
    install_guard('predict', 1)
    root = ROOT / 'adapters/reduced_papers'
    source = root / 'parallel_fold1_gpu1'
    destination = root / 'fold1'
    relative = (Path(name + '.pt'), Path('fit_dev') / (name + '.npy'), Path(name + '.receipt.json'))
    receipt = json.loads((source / relative[2]).read_text())
    assert not receipt['probe_only'] and receipt['signature']['sources'] == sources()
    assert receipt['signature']['kind'] == 'attention'
    assert receipt['signature']['seed'] == 20261109 and not receipt['signature']['probe']
    config = next(config for config in model_module().CONFIGS
                  if 'attention_' + config.family + '_' + config.variant == name)
    assert receipt['signature']['config'] == asdict(config)
    reference = json.loads((destination / 're_default.receipt.json').read_text())['signature']
    for role in ('train', 'dev'):
        assert receipt['signature'][role] == reference[role]
    assert sha(source / relative[0]) == receipt['checkpoint_sha256']
    assert sha(source / relative[1]) == receipt['prediction_sha256']
    saved = torch.load(source / relative[0], map_location='cpu', weights_only=False)
    assert saved['signature'] == receipt['signature']
    prediction = np.load(source / relative[1])
    assert prediction.shape == (2219, 12, 2) and np.isfinite(prediction).all()
    producer = json.loads((source / f'{name}.producer.json').read_text())
    assert producer['runner_sha256'] == sha(root / 'parallel_attention.py')
    assert producer['physical_gpu'] == '1' and not producer['probe_only']
    print(json.dumps({'preflight': 'passed', 'method': name, 'publish_requested': publish}), flush=True)
    if not publish:
        return
    assert process_state() not in ('T', 't', 'Z')
    temporary = []
    for path in relative:
        target = destination / path
        temp = target.with_name(target.name + f'.parallel-{os.getpid()}')
        with (source / path).open('rb') as src, temp.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        assert sha(temp) == sha(source / path)
        temporary.append(temp)
    record = {'method': name, 'pid': PID, 'start_ticks': START_TICKS,
              'publisher_sha256': sha(Path(__file__)), 'producer': producer,
              'stop_request_utc': datetime.now(timezone.utc).isoformat(), 'published': False}
    stopped = False
    committed = []
    try:
        os.kill(PID, signal.SIGSTOP)
        stopped = True
        for _ in range(100):
            if process_state() in ('T', 't'):
                break
            time.sleep(.01)
        assert process_state() in ('T', 't')
        record['stopped_observed_utc'] = datetime.now(timezone.utc).isoformat()
        lines = (root / 'evidence/train_fold1.log').read_text().splitlines()
        current = next(json.loads(line)['method'] for line in reversed(lines) if line.startswith('{"method":'))
        names = [prefix + config.family + '_' + config.variant
                 for prefix in ('', 'attention_') for config in model_module().CONFIGS]
        record['last_logged_method'] = current
        safe = names.index(current) + 1 < names.index(name)
        absent = all(not (destination / path).exists() for path in relative)
        if safe and absent:
            for path, temp in zip(relative, temporary):
                target = destination / path
                temp.replace(target)
                committed.append(target)
            assert all(sha(destination / path) == sha(source / path) for path in relative)
            record['published'] = True
            record['files'] = {str(path): sha(destination / path) for path in relative}
        else:
            record['reason'] = '이미 시작했거나 다음 설정일 가능성이 있어 게시하지 않고 별도 결과 보존'
    finally:
        try:
            if not record['published']:
                for path in reversed(committed):
                    path.unlink()
            for path in temporary:
                path.unlink(missing_ok=True)
        finally:
            try:
                if stopped:
                    os.kill(PID, signal.SIGCONT)
                    record['continue_request_utc'] = datetime.now(timezone.utc).isoformat()
            finally:
                save(source / f'{name}.publication.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
