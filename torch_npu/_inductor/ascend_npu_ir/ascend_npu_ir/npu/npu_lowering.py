import torch._ops
from torch._inductor import decomposition, lowering
from torch._inductor.fx_passes.control_dependencies import ControlDeps
from torch._inductor.lowering import lowerings, make_fallback
from torch.utils._ordered_set import OrderedSet

from torch_npu._inductor.lowering_common import (
    add_overload,
    fallback_ops_with_meta,
    resolve_op_from_name,
)

from .. import config
from ..npu.utils import get_anir_mode
from torch_npu._inductor.lowering_common import run_once
from .utils import logger

aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims


@run_once
def _register_npu_inductor_fallbacks():
    gen_set = OrderedSet()
    fallback_set = OrderedSet()
    fallback_set_exclude = OrderedSet()
    env_fallback_list = config.enable_full_lowering_fallback

    if env_fallback_list:
        for op_name in env_fallback_list.split(","):
            op_name = op_name.strip()
            op = resolve_op_from_name(op_name, logger)
            if isinstance(op, torch._ops.OpOverloadPacket):
                fallback_set.add(op)
                fallback_set_exclude.add(op)
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
    add_overload(config.FALLBACK_LIST, fallback_set)

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
                    make_fallback(op)

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
                    make_fallback(op)

    def enable_full_lowering_fallback():
        ops_to_fallback = list(
            filter(
                lambda op: op not in decomposition.decompositions
                and isinstance(
                    op,
                    (
                        torch._ops.OpOverloadPacket,
                        torch._ops.OpOverload,
                        torch._ops.HigherOrderOperator,
                    ),
                )
                and not isinstance(op, ControlDeps),
                lowerings,
            )
        )
        for op in ops_to_fallback:
            make_fallback(op)

        fallback_ops_with_meta(
            lowerings,
            decomposition.decompositions,
            make_fallback,
        )

    if config.fallback_to_aten_mode not in {"off", "include", "exclude", "all"}:
        raise AssertionError(
            f"Error! Unsupported fallback_to_aten_mode: {config.fallback_to_aten_mode} was set!"
        )

    if get_anir_mode() == "O0":
        fallback_except_gen_set(gen_set=[])
        decomposition.decompositions.clear()
        return

    if config.fallback_to_aten_mode == "include":
        fallback_via_fallback_set(fallback_set=fallback_set)
    elif config.fallback_to_aten_mode == "exclude":
        fallback_except_gen_set(gen_set=gen_set)
        fallback_via_fallback_set(fallback_set=fallback_set_exclude)
    elif config.fallback_to_aten_mode == "all":
        enable_full_lowering_fallback()
