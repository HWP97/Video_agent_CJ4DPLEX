"""동결 실행기의 장치 선택만 CPU로 바꾸는 fold 2/3 실행 경로."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
SOURCE = HERE.parents[2] / 'code/adapter_reduced.py'
SOURCE_SHA = '4de492e14114b7ac4de8116492a2a5ace274d20db8869fe7fafd368b9f1ae83c'
OLD = "probe=action=='probe';device='cpu' if probe else 'cuda'"
NEW = "probe=action=='probe';device='cpu'"


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[2] not in ('2', '3'):
        raise ValueError('CPU runner는 fold 2/3에서만 사용합니다.')
    original = SOURCE.read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA:
        raise ValueError('동결 원본 코드 SHA가 다릅니다.')
    text = original.decode()
    if text.count(OLD) != 1:
        raise ValueError('단일 장치 선택문을 확인하지 못했습니다.')
    transformed = text.replace(OLD, NEW)
    sys.path.insert(0, str(SOURCE.parent))
    namespace = {'__file__': str(SOURCE), '__name__': 'reduced_cpu_frozen'}
    exec(compile(transformed, str(SOURCE), 'exec'), namespace)
    original_sources = namespace['sources']

    def sources() -> dict[str, str]:
        return {**original_sources(), str(HERE): hashlib.sha256(HERE.read_bytes()).hexdigest(),
                'executed_cpu_source_sha256': hashlib.sha256(transformed.encode()).hexdigest()}

    namespace['sources'] = sources
    namespace['main']()


if __name__ == '__main__':
    main()
