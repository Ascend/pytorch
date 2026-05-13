import warnings
from functools import wraps

from torch_npu._init.patches.patch_manager import PatchManager


_WARN_MSG = {
    "DropoutWithByteMask": (
        "torch.nn.DropoutWithByteMask is deprecated and will be removed in future version. "
        "Use torch_npu.contrib.module.DropoutWithByteMask instead."
    ),
    "dropout_with_byte_mask": (
        "torch.nn.functional.dropout_with_byte_mask is deprecated and will be removed in future version. "
        "Use torch_npu.contrib.function.dropout_with_byte_mask instead."
    ),
}


def _wrap_torch_patch_warning_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(_WARN_MSG[func.__name__])
        return func(*args, **kwargs)

    return wrapper


@PatchManager.register_patch("warning")
def apply_npu_show_warning_patch():
    from torch_npu.utils.utils import _apply_npu_show_warning

    _apply_npu_show_warning()


@PatchManager.register_patch("warning")
def apply_deprecated_api_warning_patch():
    import torch

    torch.nn.DropoutWithByteMask = _wrap_torch_patch_warning_func(
        torch.nn.DropoutWithByteMask
    )
    torch.nn.functional.dropout_with_byte_mask = _wrap_torch_patch_warning_func(
        torch.nn.functional.dropout_with_byte_mask
    )
