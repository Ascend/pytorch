from __future__ import annotations


class FastLaunchError(RuntimeError):
    def __init__(
        self,
        reason: str,
        *,
        backend_submitted: bool,
        stable: bool = False,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.backend_submitted = bool(backend_submitted)
        self.stable = bool(stable)


class FastLaunchPlanUnavailable(RuntimeError):
    def __init__(self, reason: str, *, stable: bool = True) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stable = bool(stable)


__all__ = [
    "FastLaunchError",
    "FastLaunchPlanUnavailable",
]
