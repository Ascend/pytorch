import sys
import warnings

from torch.backends import ContextProp, PropModule
import torch_npu._C


def version():
    """Currently, the ACLNN version is not available and does not support it. 
    By default, it returns None.
    """
    warnings.warn("torch.npu.aclnn.version isn't implemented!")
    return None


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


class AclnnModule(PropModule):
    def __init__(self, m, name):
        super().__init__(m, name)

    allow_hf32 = ContextProp(_get_allow_conv_hf32, _set_allow_conv_hf32)


sys.modules[__name__] = AclnnModule(sys.modules[__name__], __name__)
allow_hf32: bool
