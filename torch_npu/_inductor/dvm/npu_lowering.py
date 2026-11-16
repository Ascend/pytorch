import logging

import torch._ops
from torch._inductor import decomposition, lowering
from torch._inductor.lowering import make_fallback
from torch.utils._ordered_set import OrderedSet

from torch_npu._inductor.lowering_common import (
    add_overload,
    resolve_op_from_name,
)

from . import config
from torch_npu._inductor.lowering_common import run_once

logger = logging.getLogger(__name__)

aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims


@run_once
def _register_npu_inductor_fallbacks():
    gen_set = OrderedSet()
    fallback_set = OrderedSet()
    env_fallback_list = config.enable_full_lowering_fallback

    if env_fallback_list:
        for op_name in env_fallback_list.split(","):
            op_name = op_name.strip()
            op = resolve_op_from_name(op_name, logger)
            if isinstance(op, torch._ops.OpOverloadPacket):
                fallback_set.add(op)
                logger.info(
                    "[npu|inductor|lowering|fallback] User specified fallback: %s",
                    op_name,
                )
            else:
                logger.warning(
                    "[npu|inductor|lowering|fallback] Cannot resolve operator: %s",
                    op_name,
                )

    add_overload(config.GENERATE_LIST, gen_set)

    def fallback_except_gen_set(gen_set):
        for op in lowering.lowerings:
            if op not in decomposition.decompositions and op not in gen_set:
                if isinstance(
                    op,
                    (
                        torch._ops.OpOverloadPacket,
                        torch._ops.OpOverload,
                        torch._ops.HigherOrderOperator,
                    ),
                ):
                    make_fallback(op, override_decomp=True)

    def fallback_via_fallback_set(fallback_set):
        for op in lowering.lowerings:
            if op not in decomposition.decompositions and op in fallback_set:
                if isinstance(
                    op,
                    (
                        torch._ops.OpOverloadPacket,
                        torch._ops.OpOverload,
                        torch._ops.HigherOrderOperator,
                    ),
                ):
                    make_fallback(op, override_decomp=True)

    # Keep GENERATE_LIST entries eligible for fusion, then apply user exclusions.
    fallback_except_gen_set(gen_set=gen_set)
    fallback_via_fallback_set(fallback_set=fallback_set)
