# Copyright (c) 2026, Huawei Technologies Co., Ltd

"""TorchNPU patches for Triton runtime imports."""

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_RUNTIME_UTILS_MODULE = "triton.backends.ascend.runtime.utils"


class _RuntimeUtilsTargetLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, source_path, finder):
        super().__init__(_RUNTIME_UTILS_MODULE, source_path)
        self._finder = finder

    def exec_module(self, module):
        try:
            super().exec_module(module)
            from triton.runtime.driver import driver

            module.target = driver.active.get_current_target()
        finally:
            self._finder.uninstall()


class _RuntimeUtilsTargetFinder(importlib.abc.MetaPathFinder):
    def __init__(self, source_path):
        self._source_path = source_path

    def find_spec(self, fullname, path=None, target=None):
        if fullname == _RUNTIME_UTILS_MODULE:
            return importlib.util.spec_from_file_location(
                fullname,
                self._source_path,
                loader=_RuntimeUtilsTargetLoader(self._source_path, self),
            )
        return None

    def uninstall(self):
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass


def patch_triton_target_no_device_init():
    """Resolve Triton's target without querying the current NPU device."""
    if _RUNTIME_UTILS_MODULE in sys.modules:
        return False
    if any(isinstance(finder, _RuntimeUtilsTargetFinder) for finder in sys.meta_path):
        return False

    try:
        import triton
    except ImportError:
        return False

    path = os.path.join(
        os.path.dirname(triton.__file__), "backends", "ascend", "runtime", "utils.py"
    )
    if not os.path.isfile(path):
        return False

    sys.meta_path.insert(0, _RuntimeUtilsTargetFinder(path))
    return True
