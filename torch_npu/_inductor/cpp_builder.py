import os
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.cpp_extension import _HERE, _TORCH_PATH, TORCH_LIB_PATH

from torch_npu.utils.cpp_extension import PYTORCH_NPU_INSTALL_PATH
from torch_npu.utils._error_code import ErrCode, pta_error

if "ASCEND_HOME_PATH" not in os.environ:
    def lazy_error():
        raise RuntimeError("Could not find ASCEND_HOME_PATH in env. Please run set_env.sh first."
                            + pta_error(ErrCode.NOT_FOUND))
    get_ascend_home = lazy_error
else:
    def get_ascend_home_from_env():
        return os.environ["ASCEND_HOME_PATH"]
    get_ascend_home = get_ascend_home_from_env

TORCH_LIB_PATH = os.path.join(_TORCH_PATH, 'lib')


def include_paths(npu: bool = False) -> List[str]:
    """
    Get the includ paths required to build a C++ extension.

    Args:
        npu: If 'True', includes NPU-specific include paths.
    
    Returns:
        A list if include path strings.
    """
    lib_include = os.path.join(_TORCH_PATH, "include")
    paths = [
        lib_include,
        # Remove this once torch/torch.h is officially no longer supported for C++ extensions.
        os.path.join(lib_include, 'torch', 'csrc', 'api', 'include'),
        # Some internal (old) Torch headers don't properly prefix their includes,
        # so we need to pass -Itorch/lib/include/TH as well.
        os.path.join(lib_include, 'TH'),
        os.path.join(lib_include, 'THC')
    ]
    if npu:
        ASCEND_HOME = get_ascend_home()
        paths.extend([
            os.path.join(ASCEND_HOME, "include"),
            os.path.join(ASCEND_HOME, "include/experiment"),
            os.path.join(ASCEND_HOME, "include/experiment/msprof"),
        ])

        paths.append(os.path.join(PYTORCH_NPU_INSTALL_PATH, "include"))
    return paths


def library_paths(npu: bool = False) -> List[str]:
    """
    Get the library paths required to build a C++.

    Args:
        npu: If 'True', includes NPU-specific library paths.

    Returns:
        A list of library path strings.
    """
    # We need to link against libtorch.so
    paths = [TORCH_LIB_PATH]
    if npu:
        if "LIBTORCH_NPU_PATH" in os.environ:
            libtorch_npu_path = os.environ["LIBTORCH_NPU_PATH"]
        else:
            libtorch_npu_path = os.path.join(PYTORCH_NPU_INSTALL_PATH, "lib")
        paths.append(libtorch_npu_path)

        ASCEND_HOME = get_ascend_home()
        cann_lib_path = os.path.join(ASCEND_HOME, "lib64")
        paths.append(cann_lib_path)

    return paths


def get_cpp_torch_device_options(
    device_type: str,
    aot_mode: bool = False,
    compile_only: bool = False,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[str]]:
    
    npu = "npu" == device_type

    definations: List[str] = []
    include_dirs: List[str] = []
    cflags: List[str] = []
    ldflags: List[str] = []
    libraries_dirs: List[str] = []
    libraries: List[str] = []
    passthough_args: List[str] = []

    include_dirs = include_paths(npu)
    libraries_dirs = library_paths(npu)

    if npu:
        definations.append("USE_NPU")
        libraries += ["torch_npu", "runtime", "ascendcl"]

        # Could not add BUILD_LIBTORCH=ON to definations because it cannot
        # process defination include "=" like -DXXX=xx.
        passthough_args += ["-DBUILD_LIBTORCH=ON -Wno-unused-function"]

    return (
        definations,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthough_args,
    )


def patch_get_cpp_torch_device_options():
    torch._inductor.cpp_builder.get_cpp_torch_device_options = get_cpp_torch_device_options