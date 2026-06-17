import logging
from collections.abc import Callable
from typing import Optional


logger = logging.getLogger(__name__)

_default_npu_flop_registry: dict[str, tuple[Callable, Optional[str]]] = {}
_npu_flop_registry: dict[str, tuple[Callable, Optional[str]]] = {}


def register_npu_flop(
    target: Optional[str] = None,
    op_name: Optional[str] = None,
    *,
    is_default: bool = False,
):
    def decorator(func: Callable) -> Callable:
        resolved_name = op_name
        if resolved_name is None:
            if target is not None and ":" in target:
                resolved_name = target.split(":", 1)[1]
            else:
                resolved_name = func.__name__
        registry = _default_npu_flop_registry if is_default else _npu_flop_registry
        if not is_default and resolved_name in registry:
            logger.error(
                "Duplicate external FLOPs registration for %s. "
                "The later registration takes precedence.",
                resolved_name,
            )
        registry[resolved_name] = (func, target)
        return func

    return decorator


def get_flop_func(op_name: str) -> Optional[Callable]:
    entry = _npu_flop_registry.get(op_name) or _default_npu_flop_registry.get(op_name)
    return entry[0] if entry else None


def get_npu_flop_targets() -> dict[str, str]:
    targets = {
        name: entry[1]
        for name, entry in _default_npu_flop_registry.items()
        if entry[1] is not None
    }
    targets.update(
        {
            name: entry[1]
            for name, entry in _npu_flop_registry.items()
            if entry[1] is not None
        }
    )
    return targets
