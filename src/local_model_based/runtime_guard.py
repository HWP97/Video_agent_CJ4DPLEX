# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# 실행: main.py가 이 경계를 통해 branch를 실행한다.
"""복사 계열 밖 원본 접근과 네트워크를 차단한다. 악성 코드용 OS sandbox는 아니다."""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys
from typing import Final
from bundle_paths import RESULTS, WORKSPACE, REPORTS

ROOT: Final = Path(__file__).resolve().parent
ENVIRONMENT: Final = Path(sys.prefix).resolve()
PACKAGE_ROOTS: Final = tuple(Path(p).resolve() for p in sys.path
                             if p and Path(p).name in {"site-packages", "dist-packages"})
WRITE_FLAGS: Final = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND


class BoundaryError(PermissionError):
    def __init__(self, operation: str, target: str) -> None:
        self.operation = operation
        self.target = target
        super().__init__(f"offline bundle boundary: {operation}: {target}")


def audit(event: str, arguments: tuple) -> None:
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind",
                 "subprocess.Popen", "os.system", "os.posix_spawn", "os.exec"}:
        raise BoundaryError(event, "network/process")
    if event == "open":
        candidate, mode, flags = arguments
        if not isinstance(candidate, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(candidate)).resolve()
        writing = bool(isinstance(mode, str) and any(c in mode for c in "wax+"))
        writing = writing or bool(isinstance(flags, int) and flags & WRITE_FLAGS)
        if writing and not path.is_relative_to(ROOT) and not path.is_relative_to(RESULTS) and path != Path("/dev/null"):
            raise BoundaryError("write_outside_bundle", str(path))
        package_file = path.is_relative_to(ENVIRONMENT) or any(path.is_relative_to(p) for p in PACKAGE_ROOTS)
        if path.is_relative_to(WORKSPACE) and not path.is_relative_to(ROOT) and not path.is_relative_to(RESULTS) and not path.is_relative_to(REPORTS) and not package_file:
            raise BoundaryError("read_original_workspace", str(path))
    if event in {"os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.link", "os.symlink", "shutil.copyfile"}:
        paths = arguments[:2] if event in {"os.rename", "os.link", "os.symlink", "shutil.copyfile"} else arguments[:1]
        for candidate in paths:
            if isinstance(candidate, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(candidate)).resolve()
                if not path.is_relative_to(ROOT) and not path.is_relative_to(RESULTS):
                    raise BoundaryError("mutate_outside_bundle", str(path))


def main() -> None:
    if len(sys.argv) < 2:
        raise BoundaryError("missing_entry", "entry.py required")
    entry = Path(sys.argv[1]).resolve()
    if not entry.is_relative_to(ROOT):
        raise BoundaryError("entry_outside_bundle", str(entry))
    sys.dont_write_bytecode = True
    sys.argv = [str(entry), *sys.argv[2:]]
    sys.path.insert(0, str(entry.parent))
    sys.path.insert(1, str(ROOT))
    sys.addaudithook(audit)
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
