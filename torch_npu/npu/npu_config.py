from logging import exception
import inspect
import os
import warnings
import torch_npu._C
from torch_npu.utils.path_manager import PathManager
from torch_npu.utils._error_code import ErrCode, pta_error, prof_error

# this file is used to enhance the npu frontend API by set_option or other.

__all__ = ["set_option", "set_aoe", 
           "set_compile_mode", "set_mm_bmm_format_nd", "get_mm_bmm_format_nd",
           "is_jit_compile_false", "finalize_dump", "init_dump", "set_dump"]

_option_map = {"ACL_PRECISION_MODE": ["allow_fp32_to_fp16", "must_keep_origin_dtype"],
               "ACL_OP_SELECT_IMPL_MODE": ["high_performance", "high_precision"],
               "ACL_AICORE_NUM": (lambda value: value.isdigit() and 1 <= int(value) <= 32),
               "ACL_OPTYPELIST_FOR_IMPLMODE": None,
               "ACL_OP_DEBUG_LEVEL": ["0", "1", "2", "3", "4"],
               "ACL_DEBUG_DIR": None,
               "ACL_OP_COMPILER_CACHE_MODE": ["disable", "enable", "force"],
               "ACL_OP_COMPILER_CACHE_DIR": None,
               "ACL_OP_DEBUG_OPTION": None}

_deprecated_option_set = {"ACL_OP_SELECT_IMPL_MODE", "ACL_OPTYPELIST_FOR_IMPLMODE"}


def _check_compile_option(name, value) -> bool:
    if name in _option_map.keys():
        if _option_map[name] is None:
            return True
        if callable(_option_map[name]):
            return _option_map[name](value)
        return value in _option_map[name]
    return True


def set_option(option):
    if not isinstance(option, dict):
        raise TypeError("npu option must be a dict." + pta_error(ErrCode.PARAM))

    if option.get("MM_BMM_ND_ENABLE") == "enable":
        set_mm_bmm_format_nd(True)
    elif option.get("MM_BMM_ND_ENABLE") == "disable":
        set_mm_bmm_format_nd(False)

    for option_name, option_value in option.items():
        if _check_compile_option(option_name, str(option_value)):
            option[option_name] = str(option_value)
        elif callable(_option_map[option_name]):
            raise ValueError(f"value of {option_name} should be in %s "
                             % (inspect.getsource(_option_map[option_name])) + f"but got {option_value}" +
                             pta_error(ErrCode.PARAM))
        else:
            raise ValueError(f"value of {option_name} should be in %s "
                             % (_option_map[option_name]) + f"but got {option_value}" +
                             pta_error(ErrCode.PARAM))
        
        if option_name in _deprecated_option_set:
            warnings.warn(f"{option_name} will be deprecated in future version. The accuracy or performance "
                          f"may not be the optimal when configuring this option. We do not recommend setting it.")

    torch_npu._C._npu_setOption(option)


def init_dump():
    option = {"mdldumpswitch": "enable"}
    torch_npu._C._npu_setOption(option)


def set_dump(cfg_file):
    if not os.path.exists(cfg_file):
        raise AssertionError("cfg_file %s path does not exists." % (cfg_file) + pta_error(ErrCode.NOT_FOUND))
    cfg_file = os.path.realpath(cfg_file)
    option = {"mdldumpconfigpath": cfg_file}
    torch_npu._C._npu_setOption(option)


def finalize_dump():
    option = {"mdldumpswitch": "disable"}
    torch_npu._C._npu_setOption(option)


def set_compile_mode(jit_compile=False):
    if torch_npu.npu.is_initialized():
        torch_npu.npu.synchronize()
    option = {"jitCompile": "enable" if jit_compile else "disable"}
    torch_npu._C._npu_setOption(option)


def set_aoe(dump_path):
    if not os.path.exists(dump_path):
        try:
            PathManager.make_dir_safety(dump_path)
        except TypeError:
            raise TypeError("Type of dump_path is invalid." + pta_error(ErrCode.TYPE)) from None
        except OSError:
            raise OSError("Value of dump_path is invalid." + pta_error(ErrCode.SYSCALL)) from None
    option = {"autotune": "enable", "autotunegraphdumppath": dump_path}
    torch_npu._C._npu_setOption(option)


"""
This global flag control mm and bmm use ND format to compute, if the flag is True,
we use ND format for mm and bmm in Linear module

useage:
```
option = {}
option["MM_BMM_ND_ENABLE"] = "enable"
torch.npu.set_option(option)
```

Default: False
"""
_MM_BMM_ND_ENABLE = True


def set_mm_bmm_format_nd(is_nd=True):
    global _MM_BMM_ND_ENABLE
    if is_nd:
        _MM_BMM_ND_ENABLE = True
    else:
        _MM_BMM_ND_ENABLE = False


def get_mm_bmm_format_nd():
    return _MM_BMM_ND_ENABLE


def is_jit_compile_false() -> bool:
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_is_jit_compile_false()


class _npuConfig:
    @classmethod
    def __setattr__(cls, name, value):
        if name == "allow_internal_format":
            option = {"ALLOW_INTERNAL_FORMAT": "enable" if value else "disable"}
            torch_npu._C._npu_setOption(option)


class _allowHF32Matmul:
    @classmethod
    def __setattr__(cls, name, value):
        if name == "allow_hf32":
            option = {"ALLOW_MATMUL_HF32": "enable" if value else "disable"}
            torch_npu._C._npu_setOption(option)

    @classmethod
    def __getattr__(cls, name):
        if name == "allow_hf32":
            hf32_value = torch_npu._C._npu_getOption("ALLOW_MATMUL_HF32")
            return hf32_value is not None and hf32_value.decode() == "enable"
        return None


class _allowHF32Conv:
    @classmethod
    def __setattr__(cls, name, value):
        if name == "allow_hf32":
            option = {"ALLOW_CONV_HF32": "enable" if value else "disable"}
            torch_npu._C._npu_setOption(option)

    @classmethod
    def __getattr__(cls, name):
        if name == "allow_hf32":
            hf32_value = torch_npu._C._npu_getOption("ALLOW_CONV_HF32")
            return (hf32_value is None) or (hf32_value.decode() == "") or (hf32_value.decode() == "enable")
        return None
