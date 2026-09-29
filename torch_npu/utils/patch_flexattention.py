from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import sys


_FLEX_MODULE = "torch.nn.attention.flex_attention"
_finder = None


def _patch_module(fa_mod):
    if getattr(fa_mod, "_npu_flex_device_patched", False):
        return

    origin_validate_device = fa_mod._validate_device

    @functools.wraps(origin_validate_device)
    def _npu_validate_device(query, key, value):
        if query.device.type != "npu":
            return origin_validate_device(query, key, value)

        if query.device != key.device or query.device != value.device:
            raise ValueError(
                "Expected query, key, and value to have the same device, "
                f"but got query.device: {query.device}, "
                f"key.device: {key.device}, "
                f"value.device: {value.device} instead."
            )

    fa_mod._validate_device = _npu_validate_device
    fa_mod._npu_flex_device_patched = True


def _remove_finder():
    global _finder

    if _finder is not None:
        try:
            sys.meta_path.remove(_finder)
        except ValueError:
            pass
        _finder = None


class _FlexLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        create_module = getattr(self.loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module):
        self.loader.exec_module(module)
        _patch_module(module)
        _remove_finder()


class _FlexFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _FLEX_MODULE or fullname in sys.modules:
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _FlexLoader(spec.loader)
        return spec


def _patch_flex_attention_device():
    """Install lazy FlexAttention device validation patch."""
    global _finder

    module = sys.modules.get(_FLEX_MODULE)

    # 用户已经提前导入 FlexAttention 的边界场景
    if module is not None:
        _patch_module(module)
        return

    # import torch_npu 时不主动导入 Flex_attention
    if _finder is None:
        _finder = _FlexFinder()
        sys.meta_path.insert(0, _finder)
