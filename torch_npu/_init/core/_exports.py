from functools import wraps
from importlib import import_module

import torch

import torch_npu


_LAZY_PYTHON_SYMBOLS = {
    "HiFloat8Tensor": ("torch_npu.utils.hif8_tensor", "_HiFloat8Tensor"),
    "erase_stream": ("torch_npu.npu.utils", "_erase_stream"),
    "matmul_checksum": ("torch_npu.asd.checksum", "_matmul_checksum"),
}


def _append_unique(all_list, names):
    for name in names:
        all_list.append(name)


def _export_npu_ops(globals_dict, all_list):
    """
    Export NPU custom ops from torch.ops.npu.
    Rules:
        - torch.ops.npu.<op_name> -> torch_npu.<op_name>
        - torch.ops.npu.<op_name> -> torch.<op_name> deprecated wrapper
    """

    def _wrap_torch_error_func(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # lazy import to avoid early utils import / circular dependency.
            from torch_npu.utils._error_code import ErrCode, pta_error

            raise RuntimeError(
                f"torch.{func.__name__} is deprecated and will be removed in future version. "
                f"Use torch_npu.{func.__name__} instead."
                + pta_error(ErrCode.NOT_SUPPORT)
            )

        return wrapper

    from torch_npu.utils.exposed_api import public_npu_functions

    for name in dir(torch.ops.npu):
        if name.startswith("__") or name in ["_dir", "name"]:
            continue
        globals_dict[name] = getattr(torch.ops.npu, name)
        if name in public_npu_functions and name not in all_list:
            _append_unique(all_list, [name])
        setattr(torch, name, _wrap_torch_error_func(getattr(torch.ops.npu, name)))


def _export_dtype_symbols():
    """
    Export DType symbols to from C extension.
    Rule:
        - torch_npu._C._cd.DType.<dtype_name> -> torch_npu.<dtype_name>
    """
    for name in dir(torch_npu._C._cd.DType):
        if name.startswith("__") or name in ["_dir", "name"]:
            continue
        setattr(torch_npu, name, getattr(torch_npu._C._cd.DType, name))


def _export_lazy_python_apis(globals_dict, all_list):
    """
    Lazily export Python-defined top-levels APIs.
    Excample:
        torch_npu.utils.hif8_tensor._HiFloat8Tensor -> torch_npu.HiFloat8Tensor
    """
    module_name = globals_dict.get("__name__", "torch_npu")

    def _lazy_import_api(name: str):
        try:
            import_path, attr_name = _LAZY_PYTHON_SYMBOLS[name]
        except KeyError as exc:
            raise AttributeError(
                f"module {module_name!r} has no attribute {name!r}"
            ) from exc

        module = import_module(import_path)
        value = getattr(module, attr_name)
        globals_dict[name] = value
        return value

    def __getattr__(name: str):
        return _lazy_import_api(name)

    def __dir__():
        return sorted(set(globals_dict) | set(_LAZY_PYTHON_SYMBOLS))

    globals_dict["__getattr__"] = __getattr__
    globals_dict["__dir__"] = __dir__
    _append_unique(all_list, _LAZY_PYTHON_SYMBOLS.keys())


def _export_public_apis():
    """
    Export torch_npu public APIs.
    1. python APIs:
        - torch_npu.utils.hif8_tensor._HiFloat8Tensor -> torch_npu.HiFloat8Tensor
        - torch_npu.npu.utils._erase_stream           -> torch_npu.erase_stream
        - torch_npu.asd.checksum._matmul_checksum     -> torch_npu.matmul_checksum

    2. NPU custom ops:
        - torch.ops.npu.<op_name> -> torch_npu.<op_name>
        - torch.ops.npu.<op_name> -> torch.<op_name> deprecated wrapper

    3. DType symbols:
        - torch_npu._C._cd.DType.<dtype_name> -> torch_npu.<dtype_name>
    """

    _export_dtype_symbols()
    import torch_npu as _torch_npu

    globals_dict = _torch_npu.__dict__
    all_list = _torch_npu.__all__

    _export_lazy_python_apis(globals_dict, all_list)
    _export_npu_ops(globals_dict, all_list)
    _export_dtype_symbols()
