# Copyright (c) 2026, Huawei Technologies Co., Ltd.
"""Install the generated NPU schemas in Inductor's AOTI registry."""


_registered = False


def register_npu_aoti_fallbacks():
    """Merge the NPU-generated fallback schemas into torchgen's registry.

    ``torchnpugen`` owns the NPU additions (including ``aten.any.dims``), and
    ``torchgen`` owns the common AOTI registry consumed by C++ wrapper codegen.
    Neither import depends on the default torch_npu Inductor backend.
    """
    import torchgen.aoti.fallback_ops as fallback_ops
    from torchnpugen.aoti.fallback_ops import (
        inductor_fallback_ops_npu,
        inductor_fallback_ops_npu_not_support,
    )

    global _registered
    if _registered:
        return
    fallback_ops.inductor_fallback_ops.update(inductor_fallback_ops_npu)
    for key in inductor_fallback_ops_npu_not_support:
        fallback_ops.inductor_fallback_ops.pop(key, None)
    _registered = True
