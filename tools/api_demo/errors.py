"""데모 자산 준비·재생 경계의 오류."""


class DemoError(RuntimeError):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)
