import os
from typing import Callable

from torch.utils._ordered_set import OrderedSet

from ...config import (
    enable_fused_matmul_relu,
    enable_grouped_matmul_fusion,
    enable_multi_slice_concat,
    log,
)
from ..utils.fx_pass_level import FxPassLevel, PassType


ASCEND_CUSTOME_PASS_REGISTER = {
    pass_type: {level: [] for level in FxPassLevel} for pass_type in PassType
}

# Passes that stay off until their config switch is set. A disabled pass is never
# registered, so neither the inference nor the training path runs it.
DEFAULT_SHUT_DOWN_PASSES = (
    (() if enable_fused_matmul_relu else ("fused_matmul_relu_pass",))
    + (() if enable_multi_slice_concat else ("multi_slice_concat_pass",))
    + (() if enable_grouped_matmul_fusion else ("grouped_matmul_fusion_pass",))
)


def _get_shut_down_pass_set():
    """Collect the passes to skip: the default off list plus SHUT_DOWN_FX_PASS_LIST."""
    shut_down_str = os.environ.get("SHUT_DOWN_FX_PASS_LIST", "")
    env_list = [p.strip() for p in shut_down_str.split(",") if p.strip()]
    return OrderedSet(DEFAULT_SHUT_DOWN_PASSES + tuple(env_list))


def register_custom_pass(
    pass_type: int = PassType.PRE,
    fx_pass_level: int = FxPassLevel.LEVEL1,
    ignore_inference_check: bool = False,
):
    """Register a custom fx pass.

    ignore_inference_check: the pass only does equivalent rewrites and holds on
    training graphs too, so the driver need not gate it behind is_inference_check.
    The mark lives on the function rather than in a name list inside the driver,
    so renaming a pass cannot silently disable it.
    """

    def decorator(fn: Callable) -> Callable:
        fn.ignore_inference_check = ignore_inference_check
        # default passes never collide on name
        # merge the built-in default off list with SHUT_DOWN_FX_PASS_LIST
        shut_down_set = _get_shut_down_pass_set()
        # skip registration if the function name is in the off list (pass disabled)
        if "all" in shut_down_set:
            log.debug("Ignoring all registration in graph optimizer.")
            return fn
        elif fn.__name__ in shut_down_set:
            log.debug("Ignoring registration of %s", fn.__name__)
            return fn  # return the original function without registering
        else:
            # otherwise register normally
            log.debug(
                "Registering function %s from module %s with pass_type=%s, fx_pass_level=%s",
                fn.__name__,
                fn.__module__,
                pass_type,
                fx_pass_level,
            )
            ASCEND_CUSTOME_PASS_REGISTER[pass_type][fx_pass_level].append(fn)
            return fn

    return decorator
