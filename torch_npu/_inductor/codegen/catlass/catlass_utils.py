import functools
import logging
import copy
import os
import sys
from typing import Any, List, Optional

import torch
from torch._inductor.runtime.runtime_utils import cache_dir
from torch_npu.npu import matmul

from ... import config as npu_config

log = logging.getLogger("torch._inductor")


def get_npu_arch() -> Optional[str]:
    try:
        npu_arch = npu_config.target.arch
        return npu_arch
    except Exception as e:
        log.error("Error getting npu arch: %s", e)
        return None


@functools.lru_cache(None)
def try_import_catlass() -> bool:
    """
    We want to support three ways of passing in CATLASS:
    1. fbcode, handled by the internal build system.
    2. pip install catlass, which provides the catlass_cppgen package.
    3. User specifies catlass_dir. The default is ../third_party/catlass/,
       which is the directory when developers build from source.
    """

    try:
        import catlass  # type: ignore[import-not-found]
        import catlass_cppgen  # type: ignore[import-not-found]

        log.debug(
            "Found catlass_cppgen in python search path, overriding npu_config.catlass.catlass_dir"
        )
        catlass_library_dir = os.path.dirname(catlass_cppgen.__file__)
        assert os.path.isdir(catlass_library_dir), (
            f"{catlass_library_dir} is not a directory"
        )
        npu_config.catlass.catlass_dir = os.path.abspath(
            os.path.join(
                catlass_library_dir,
                "source",
            )
        )
        return True
    except ModuleNotFoundError:
        log.debug(
            "catlass_cppgen not found in sys.path, trying to import from npu_config.catlass.catlass_dir"
        )

    # Copy CATLASS python scripts to a temp dir and add the temp dir to Python search path.
    # This is a temporary hack to avoid CATLASS module naming conflicts.

    catlass_py_full_path = os.path.abspath(
        os.path.join(npu_config.catlass.catlass_dir, "python/catlass_cppgen")
    )
    tmp_catlass_py_full_path = os.path.abspath(
        os.path.join(cache_dir(), "torch_catlass_library")
    )
    dst_link = os.path.join(tmp_catlass_py_full_path, "catlass_cppgen")

    if os.path.isdir(catlass_py_full_path):
        if tmp_catlass_py_full_path not in sys.path:
            if os.path.exists(dst_link):
                assert os.path.islink(dst_link), (
                    f"{dst_link} is not a symlink. Try to remove {dst_link} manually and try again."
                )
                assert os.path.realpath(os.readlink(dst_link)) == os.path.realpath(
                    catlass_py_full_path
                ), f"Symlink at {dst_link} does not point to {catlass_py_full_path}"
            else:
                os.makedirs(tmp_catlass_py_full_path, exist_ok=True)
                os.symlink(catlass_py_full_path, dst_link)
            sys.path.append(dst_link)
        try:
            import catlass_cppgen.op
            import catlass_cppgen.common
            import catlass_cppgen.catlass
            import catlass_cppgen.kernel

            return True
        except ImportError as e:
            log.debug(
                "Failed to import CATLASS packages: %s, ignoring the CATLASS backend.",
                str(e),
            )
    else:
        log.debug(
            "Failed to import CATLASS packages: CATLASS repo does not exist: %s",
            catlass_py_full_path,
        )
    return False


def _normalize_npu_arch(arch: str) -> str:
    if "910B" in arch or arch.startswith("Ascend910_93"):
        return "910B"
    elif arch.startswith("Ascend910_95"):
        return "910D"
    else:
        raise NotImplementedError(f"Unsupported npu arch: {arch}")


def _normalize_npu_arch_to_atlas(arch: str) -> str:
    from catlass_cppgen.catlass.arch.arch import Arch
    if "910B" in arch or arch.startswith("Ascend910_93"):
        return Arch.AtlasA2
    elif arch.startswith("Ascend910_95"):
        return Arch.AtlasA5
    else:
        raise NotImplementedError(f"Unsupported npu arch: {arch}")


def _catlass_tensor_from_node(node):
    from catlass_cppgen.common.op_tensor import OpTensor
    from catlass_cppgen.common.data_type import DataType

    if not node:
        return None
    try:
        shape = tuple(node.data.get_size())
        stride = tuple(node.data.get_stride())
    except AttributeError:
        shape = tuple(node.get_size())
        stride = tuple(node.get_stride())
    element = DataType.from_dtype(node.get_dtype())
    return OpTensor.from_shape_stride(
        shape=shape,
        stride=stride,
        dtype=element,
    )


@functools.lru_cache(None)
def _gen_ops_cached(arch: str, op_tensors=None, is_group_mm=False) -> List[Any]:
    from catlass_cppgen.op.gemm import Gemm
    from catlass_cppgen.op.group_gemm import GroupGemm
    from catlass_cppgen.common.data_type import DataType
    from .catlass_library.gemm_autotune import generate_configs

    arch = _normalize_npu_arch_to_atlas(arch)

    ops = []

    use_hf32 = False
    if (matmul.allow_hf32 and op_tensors[0].dtype == DataType.FLOAT and op_tensors[1].dtype == DataType.FLOAT):
        use_hf32 = True

    element_C = op_tensors[0].dtype
    if len(op_tensors) == 2:
        gemm_plan = Gemm(atlas_arch=arch, element_C=element_C, A=op_tensors[0], B=op_tensors[1])
        kernels = gemm_plan.get_kernels()
    elif not is_group_mm: # mm with bias
        gemm_plan = Gemm(atlas_arch=arch, element_C=element_C, A=op_tensors[0], B=op_tensors[1], Bias=op_tensors[2])
        kernels = gemm_plan.get_kernels()
    else: # group mm
        gemm_plan = GroupGemm(atlas_arch=arch, element_C=element_C, A=op_tensors[0], B=op_tensors[1], 
                              groupList=op_tensors[2])
        kernels = gemm_plan.get_kernels()
    
    for kernel in kernels:
        tilings = generate_configs(arch, kernel)
        if tilings:
            for tiling in tilings:
                k = copy.deepcopy(kernel)
                k.tune(tiling[0], tiling[1], is_hf32=use_hf32)
                ops.append(k)
        else:
            kernel.tune(is_hf32=use_hf32)
            ops.append(kernel)

    if arch is None:
        log.error(
            "Cannot detect npu arch %s. "
            "Will discard all catlass ops. "
            "Please consider setting _inductor.npu.arch configs.",
            arch,
        )
        return []
    return ops


def gen_ops(op_tensors=None, is_group_mm=False) -> List[Any]:
    """
    Generates all supported CATLASS Gemm operations for M, N, K
    """
    arch = get_npu_arch()
    return _gen_ops_cached(arch, tuple(op_tensors), is_group_mm)