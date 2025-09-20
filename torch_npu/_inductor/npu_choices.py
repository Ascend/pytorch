import typing
from typing import Any, Dict, List, Type, TYPE_CHECKING
import sympy
from torch._inductor import config
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.runtime.hints import ReductionHint
from torch._inductor.virtualized import V


@staticmethod
def should_use_persistent_reduction(
        features: SIMDKernelFeatures, cooperative_reduction: bool
) -> bool:
    """
    Heuristic to decide if a persistent reduction should be used.
    """
    if not config.triton.persistent_reductions:
        return False
    threshold = {
        ReductionHint.INNER: 1024,
        ReductionHint.DEFAULT: 1024
    }.get(features.get_reduction_hint(), 64)
    if cooperative_reduction:
        # The RSPLIT of cooperative reductions means each thread block is operating on fewer elements
        try:
            threshold *= 32 // min(V.graph.sizevars.size_hint(features.numel), 32)
        except ValueError:
            pass  # unbacked symint

    if config.triton.multi_kernel:
        threshold *= 16
    return V.graph.sizevars.statically_known_leq(features.reduction_numel, threshold)  # type: ignore[arg-types]
