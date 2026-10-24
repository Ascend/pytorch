import os
from torch.utils.cpp_extension import _TORCH_PATH

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


def include_paths(npu: bool = False) -> list[str]:
    """
    Get the include paths required to build a C++ extension.

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
    include_path = os.path.join(PYTORCH_NPU_INSTALL_PATH, "include")
    paths.extend([include_path])
    if npu:
        ASCEND_HOME = get_ascend_home()
        paths.extend([
            os.path.join(ASCEND_HOME, "include"),
            os.path.join(ASCEND_HOME, "pkg_inc"),
            os.path.join(ASCEND_HOME, "pkg_inc/profiling"),
            os.path.join(ASCEND_HOME, "include/experiment"),
            os.path.join(ASCEND_HOME, "include/experiment/msprof"),
        ])

    return paths


def library_paths(npu: bool = False) -> list[str]:
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


def patch_get_cpp_torch_device_options():
    # Keep the legacy entry point used by AscendC; delegate to the compat layer.
    from torch_npu._compat.inductor import patch_npu_cpp_device_options

    patch_npu_cpp_device_options()
