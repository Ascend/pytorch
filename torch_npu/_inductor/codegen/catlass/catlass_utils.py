import functools
import logging
from typing import Any, List, Optional, Tuple, Union

import torch

from . import catlass_library


log = logging.getLogger("torch._inductor")


def get_npu_arch() -> Optional[str]:
    try:
        from ... import config as npu_config

        npu_arch = npu_config.target.arch
        return npu_arch
    except Exception as e:
        log.error("Error getting npu arch: %s", e)
        return None


def _normalize_npu_arch(arch: str) -> str:
    if "910B" in arch or arch.startswith("Ascend910_93"):
        return "910B"
    elif arch.startswith("Ascend910_95"):
        return "910D"
    else:
        raise NotImplementedError(f"Unsupported npu arch: {arch}")


@functools.lru_cache(None)
def _gen_ops_cached(arch: str, shape_desc=None) -> List[Any]:
    from .catlass_library import generator as catlass_generator
    from .catlass_library.manifest import manifest

    if arch is None:
        log.error(
            "Cannot detect npu arch %s. "
            "Will discard all catlass ops. "
            "Please consider setting _inductor.npu.arch configs.",
            arch,
        )
        return []

    arch = _normalize_npu_arch(arch)

    try:
        func = getattr(catlass_generator, "Generate" + arch)
        func(manifest, shape_desc)
    except AttributeError as e:
        raise NotImplementedError(
            "Arch " + arch + " is not supported by current catlass lib."
        ) from e

    return manifest.get_ops(shape_desc)


def gen_ops(shape_desc=None) -> List[Any]:
    """
    Generates all supported CATLASS Gemm operations for M, N, K
    """
    arch = get_npu_arch()
    return _gen_ops_cached(arch, shape_desc)


def torch_dtype_to_catlass_type(
    torch_dtype: torch.dtype,
) -> "catlass_library.library.DataType":  # type: ignore[name-defined] # noqa: F821
    if torch_dtype == torch.float:
        return catlass_library.library.DataType.f32
    elif torch_dtype == torch.half:
        return catlass_library.library.DataType.f16
    elif torch_dtype == torch.bfloat16:
        return catlass_library.library.DataType.bf16
    elif torch_dtype == torch.uint8:
        return catlass_library.library.DataType.u8
    elif torch_dtype == torch.uint16:
        return catlass_library.library.DataType.u16
    elif torch_dtype == torch.uint32:
        return catlass_library.library.DataType.u32
    elif torch_dtype == torch.uint64:
        return catlass_library.library.DataType.u64
    elif torch_dtype == torch.int8:
        return catlass_library.library.DataType.s8
    elif torch_dtype == torch.int16:
        return catlass_library.library.DataType.s16
    elif torch_dtype == torch.int32:
        return catlass_library.library.DataType.s32
    elif torch_dtype == torch.int64:
        return catlass_library.library.DataType.s64
    else:
        raise NotImplementedError(f"Unsupported data type: {torch_dtype=}")


def dtype_match(
    torch_dtype: Optional[torch.dtype],
    catlass_dtype: "catlass_library.library.DataType",  # type: ignore[name-defined]  # noqa: F821
) -> bool:
    if torch_dtype == torch.float:
        return catlass_dtype == catlass_library.library.DataType.f32
    elif torch_dtype == torch.half:
        return catlass_dtype == catlass_library.library.DataType.f16
    elif torch_dtype == torch.bfloat16:
        return catlass_dtype == catlass_library.library.DataType.bf16
    elif torch_dtype == torch.int8:
        return catlass_dtype == catlass_library.library.DataType.s8
    elif torch.dtype == torch.uint8:
        return catlass_dtype == catlass_library.library.DataType.u8
    elif torch.dtype == torch.int32:
        return catlass_dtype == catlass_library.library.DataType.s32
    else:
        return False


def get_accumulator_dtype(
    input_torch_dtypes: List[torch.dtype],
) -> Optional[torch.dtype]:
    """
    Given a pair of input torch dtypes, returns the inferred accumulator torch dtype.
    """

    if len(input_torch_dtypes) != 2:
        return None

    torch_dtype = None
    if input_torch_dtypes[0] == input_torch_dtypes[1]:
        torch_dtype = input_torch_dtypes[0]

    if torch_dtype in {torch.half, torch.bfloat16, torch.float}:
        return torch.float
    raise NotImplementedError(f"Unsupported data types: {input_torch_dtypes=}")
