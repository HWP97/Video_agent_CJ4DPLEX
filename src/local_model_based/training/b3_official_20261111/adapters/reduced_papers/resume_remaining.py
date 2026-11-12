"""중단 증거를 보존하고 동결 실행기로 잔여 작업만 수행한다."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'adapters/reduced_papers'
sys.path.insert(0, str(ROOT / 'code'))
from adapter_io import sha
from adapter_reduced import save


def inactive() -> None:
    for proc in Path('/proc').iterdir():
        if proc.name.isdecimal() and int(proc.name) != os.getpid():
            try:
                args = (proc / 'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            assert not any(arg.endswith((b'/adapter_reduced.py', b'/parallel_attention.py')) for arg in args), args


def publish(name: str) -> None:
    import publish_parallel
    original = sys.argv
    sys.argv = [str(HERE / 'publish_parallel.py'), name]
    try:
        publish_parallel.main()
    finally:
        sys.argv = original
    source = HERE / 'parallel_fold1_gpu1'
    destination = HERE / 'fold1'
    paths = [Path(name + '.pt'), Path('fit_dev') / (name + '.npy'),
             Path(name + '.producer.json'), Path(name + '.receipt.json')]
    assert all(not (destination / path).exists() for path in paths)
    for path in paths:
        with (source / path).open('rb') as src, (destination / path).open('xb') as dst:
            shutil.copyfileobj(src, dst)
        assert sha(source / path) == sha(destination / path)
    save(source / f'{name}.offline_publication.json', {
        'files': {str(path): sha(destination / path) for path in paths},
        'time_unix': time.time(), 'no_same_config_fit_running': True, 'API_calls': 0})


def prepare() -> None:
    inactive()
    archive = HERE / 'interrupted_20261111_resume'
    archive.mkdir(exist_ok=False)
    files = {}
    for relative in ('fold1/moif_no_direction.pt', 'parallel_fold1_gpu1/attention_moif_first_only.pt'):
        source = HERE / relative
        assert not source.with_suffix('.receipt.json').exists()
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(source)
        shutil.copy2(source, target)
        assert sha(target) == digest
        source.unlink()
        files[relative] = digest
    for name in ('train_fold1.log', 'parallel_fold1_gpu1.log'):
        shutil.copy2(HERE / 'evidence' / name, archive / name)
    save(archive / 'receipt.json', {'preserved_files': files, 'time_unix': time.time(), 'API_calls': 0})
    publish('attention_moif_default')


def train() -> None:
    import adapter_reduced
    original_fit = adapter_reduced.fit

    def fit(config, kind, fold, probe, training, development, folder, device):
        name = ('attention_' if kind == 'attention' else '') + config.family + '_' + config.variant
        if kind == 'attention' and config.family == 'moif' and not (folder / (name + '.receipt.json')).exists():
            receipt = HERE / 'parallel_fold1_gpu1' / (name + '.producer.json')
            deadline = time.monotonic() + 14400
            while not receipt.exists():
                assert time.monotonic() < deadline, name
                time.sleep(5)
            publish(name)
        return original_fit(config, kind, fold, probe, training, development, folder, device)

    adapter_reduced.fit = fit
    sys.argv = [str(ROOT / 'code/adapter_reduced.py'), 'train', '1']
    adapter_reduced.main()


if __name__ == '__main__':
    match sys.argv[1]:
        case 'prepare':
            prepare()
        case 'train':
            train()
        case unknown:
            raise RuntimeError(unknown)
