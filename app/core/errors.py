from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ServiceErrorDetail:
    code: str
    message: str
    service: str
    model: Optional[str] = None


class BackendServiceError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        service: str,
        model: Optional[str] = None,
        status_code: int = 503,
    ):
        super().__init__(message)
        self.detail = ServiceErrorDetail(code=code, message=message, service=service, model=model)
        self.status_code = status_code
