__all__ = [
    "version",
    "allow_hf32",
]

import sys

from contextlib import contextmanager
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule

import torch_npu._C
from .backends import version


def _set_allow_conv_hf32(_enabled: bool):
    r"""Set the device supports conv operation hf32.
    Args:
        Switch for hf32.
    """
    option = {"ALLOW_CONV_HF32": "enable" if _enabled else "disable"}
    torch_npu._C._npu_setOption(option)


def _get_allow_conv_hf32() -> bool:
    r"""Return the device supports conv operation hf32 is enabled or not.
    """
    hf32_value = torch_npu._C._npu_getOption("ALLOW_CONV_HF32")
    return (hf32_value is None) or (hf32_value.decode() == "") or (hf32_value.decode() == "enable")


def set_flags(_allow_hf32=None, ):
    orig_flags = (_get_allow_conv_hf32(),)
    if _allow_hf32 is not None:
        _set_allow_conv_hf32(_allow_hf32)
    return orig_flags


@contextmanager
def flags(allow_hf32=True, ):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(allow_hf32)
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


class AclnnModule(PropModule):
    def __init__(self, m, name):
        super().__init__(m, name)

    allow_hf32 = ContextProp(_get_allow_conv_hf32, _set_allow_conv_hf32)


sys.modules[__name__] = AclnnModule(sys.modules[__name__], __name__)
allow_hf32: bool
