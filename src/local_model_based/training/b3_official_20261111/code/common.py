from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryError(Exception):
    reason: str
    detail: str = ''

