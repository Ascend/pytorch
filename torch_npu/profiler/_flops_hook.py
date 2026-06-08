import functools
import importlib
import logging
import sys
import threading
from collections.abc import Callable
from typing import Any

from torch_npu.npu.mstx import mstx

from ._flops_registry import get_flop_func, get_npu_flop_targets


logger = logging.getLogger(__name__)

_FLOPS_DOMAIN = "mfu_flops"


def _resolve_target(target: str):
    if ":" not in target:
        logger.warning("Invalid FLOPs target format: %s", target)
        return None, None
    module_path, attr_path = target.split(":", 1)
    try:
        module_obj = importlib.import_module(module_path)
    except ImportError:
        logger.warning("Cannot import FLOPs target module: %s", module_path)
        return None, None

    obj = module_obj
    attrs = attr_path.split(".")
    for attr in attrs[:-1]:
        if not hasattr(obj, attr):
            logger.warning("Cannot resolve FLOPs target: %s", target)
            return None, None
        obj = getattr(obj, attr)
    return obj, attrs[-1]


def _build_target_ops() -> dict[str, tuple[Any, str]]:
    importlib.import_module("torch_npu.profiler._flops_formulas")

    target_ops = {}
    for op_name, target in get_npu_flop_targets().items():
        parent, attr_name = _resolve_target(target)
        if parent is not None and attr_name is not None:
            target_ops[op_name] = (parent, attr_name)
    return target_ops


def _find_existing_refs(original_func: Callable, attr_name: str):
    refs = []
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            ref = getattr(module, attr_name, None)
        except Exception:
            continue
        if ref is original_func:
            refs.append((module, attr_name))
    return refs


class FlopsHookManager:
    _local = threading.local()
    _original_funcs: dict[str, Callable] = {}
    _patched_targets: dict[str, tuple[Any, str]] = {}
    _extra_refs: dict[str, list[tuple[Any, str]]] = {}
    _installed = False

    @classmethod
    def install(cls, target_ops: dict[str, tuple[Any, str]] | None = None):
        if cls._installed:
            return
        if target_ops is None:
            target_ops = _build_target_ops()

        try:
            for op_name, (module_obj, attr_name) in target_ops.items():
                original = getattr(module_obj, attr_name, None)
                if original is None:
                    logger.warning("Cannot find FLOPs target %s.%s", module_obj, attr_name)
                    continue
                wrapped = cls._make_wrapper(op_name, original)
                setattr(module_obj, attr_name, wrapped)
                cls._original_funcs[op_name] = original
                cls._patched_targets[op_name] = (module_obj, attr_name)
                cls._extra_refs[op_name] = []
                for ref_module, ref_attr in _find_existing_refs(original, attr_name):
                    if ref_module is module_obj and ref_attr == attr_name:
                        continue
                    setattr(ref_module, ref_attr, wrapped)
                    cls._extra_refs[op_name].append((ref_module, ref_attr))
        except Exception:
            cls._restore()
            raise

        cls._installed = bool(cls._patched_targets)

    @classmethod
    def uninstall(cls):
        cls._restore()

    @classmethod
    def _restore(cls):
        for op_name, (module_obj, attr_name) in cls._patched_targets.items():
            original = cls._original_funcs.get(op_name)
            if original is not None:
                setattr(module_obj, attr_name, original)
        for op_name, refs in cls._extra_refs.items():
            original = cls._original_funcs.get(op_name)
            if original is not None:
                for ref_module, ref_attr in refs:
                    setattr(ref_module, ref_attr, original)
        cls._original_funcs.clear()
        cls._patched_targets.clear()
        cls._extra_refs.clear()
        cls._installed = False

    @classmethod
    def is_installed(cls) -> bool:
        return cls._installed

    @classmethod
    def _make_wrapper(cls, op_name: str, original_func: Callable) -> Callable:
        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            if getattr(cls._local, "in_hook", False):
                return original_func(*args, **kwargs)

            cls._local.in_hook = True
            range_id = None
            try:
                flop_func = get_flop_func(op_name)
                if flop_func is not None:
                    try:
                        flops = flop_func(*args, **kwargs)
                        if flops is not None and flops >= 0:
                            range_id = mstx.range_start(
                                f"{flops}-{op_name}", domain=_FLOPS_DOMAIN
                            )
                    except Exception:
                        logger.warning(
                            "Failed to calculate FLOPs for %s", op_name, exc_info=True
                        )
                return original_func(*args, **kwargs)
            finally:
                if isinstance(range_id, int) and range_id > 0:
                    mstx.range_end(range_id, domain=_FLOPS_DOMAIN)
                cls._local.in_hook = False

        return wrapper
