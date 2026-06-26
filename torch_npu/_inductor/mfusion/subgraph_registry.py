import logging
import threading
from dataclasses import dataclass
from typing import Optional

import torch.fx


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Payload:
    fx_gm: torch.fx.GraphModule
    is_dynamic: bool
    torch_name: Optional[str] = None
    full_op_name: Optional[str] = None


_registry: dict[str, Payload] = {}
_lock = threading.RLock()


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"invalid subgraph_name: {name!r}")


def _validate_payload(payload: Payload) -> None:
    if not isinstance(payload.fx_gm, torch.fx.GraphModule):
        raise TypeError("payload.fx_gm must be torch.fx.GraphModule")
    if not isinstance(payload.is_dynamic, bool):
        raise TypeError("payload.is_dynamic must be bool")


def register(name: str, payload: Payload) -> None:
    _validate_name(name)
    _validate_payload(payload)
    with _lock:
        _registry[name] = payload


def get(name: str) -> Payload:
    _validate_name(name)
    with _lock:
        payload = _registry.get(name)
        if payload is None:
            raise KeyError(f"mfusion payload not found for subgraph_name: {name!r}")
        return payload


def pop(name: str) -> Payload:
    _validate_name(name)
    with _lock:
        payload = _registry.pop(name, None)
        if payload is None:
            raise KeyError(f"mfusion payload not found for subgraph_name: {name!r}")
        logger.debug("pop subgraph_name: %s", name)
        return payload


def print_registry() -> None:
    with _lock:
        for name, payload in _registry.items():
            logger.debug("subgraph_name: %s, payload: %s", name, payload)
