import sys
import types

import torch

from torch_npu._init.patches.patch_manager import PatchManager
from torch_npu.contrib.function import npu_functional
from torch_npu.contrib.module import npu_modules
from torch_npu.utils._error_code import ErrCode, pta_error


all_monkey_patches = [
    ["nn.functional", npu_functional],
    ["nn", npu_modules],
]


def _apply_patches(monkey_patches):
    def _getattr(module_list, root_module=torch):
        if len(module_list) <= 1:
            return root_module

        if hasattr(root_module, module_list[0]):
            return _getattr(module_list[1:], getattr(root_module, module_list[0]))

        empty_module_name = f"{root_module.__name__}.{module_list[0]}"
        sys.modules[empty_module_name] = types.ModuleType(empty_module_name)
        setattr(root_module, module_list[0], sys.modules.get(empty_module_name))
        return _getattr(module_list[1:], getattr(root_module, module_list[0]))

    for dest, patch in monkey_patches:
        dest_module = _getattr(dest.split("."), root_module=torch)
        last_module_level = dest.split(".")[-1]

        if not isinstance(patch, types.ModuleType):
            setattr(dest_module, last_module_level, patch)
            continue

        if not hasattr(dest_module, last_module_level) or not hasattr(patch, "__all__"):
            setattr(dest_module, last_module_level, patch)
            sys.modules[f"{dest_module.__name__}.{last_module_level}"] = patch
            continue

        if not hasattr(patch, "__all__"):
            raise NotImplementedError(
                "Patch module must have __all__ definition."
                + pta_error(ErrCode.NOT_SUPPORT)
            )

        dest_module = getattr(dest_module, last_module_level)
        for attr in patch.__all__:
            setattr(dest_module, attr, getattr(patch, attr))


@PatchManager.register_patch("monkey")
def apply_monkey_patches():
    _apply_patches(all_monkey_patches)
