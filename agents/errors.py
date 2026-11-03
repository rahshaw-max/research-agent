"""Pipeline-level exception for stage failures."""
from __future__ import annotations


class StageFailed(Exception):
    """Raised when a pipeline stage fails in a way that cannot be silently degraded.

    Attributes:
        stage:  Name of the failed stage (e.g. "researcher", "architect_v1").
        reason: Human-readable failure description.
        cause:  Original exception, if any.
    """

    def __init__(self, stage: str, reason: str, cause: Exception | None = None) -> None:
        super().__init__(f"[{stage}] {reason}")
        self.stage = stage
        self.reason = reason
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause
