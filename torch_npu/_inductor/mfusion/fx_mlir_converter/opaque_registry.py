import itertools
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Payload:
    target: Any
    args_spec: Any
    kwargs_spec: Any
    meta: dict[str, Any]
    result_paths: tuple[tuple[int, ...], ...] = ()


_registry: dict[str, Payload] = {}
_counter = itertools.count()
_lock = threading.RLock()


def new_target_name() -> str:
    with _lock:
        return f"torch.mfusion_opaque.call_{next(_counter)}"


def register(target_name: str, payload: Payload) -> None:
    with _lock:
        _registry[target_name] = payload


def pop(target_name: str) -> Payload:
    with _lock:
        payload = _registry.pop(target_name, None)
        if payload is None:
            raise KeyError(f"opaque mfusion payload not found: {target_name!r}")
        return payload
