from torch._inductor import lowering
from torch._inductor.lowering import lowerings, make_fallback
from torch._inductor import decomposition
import torch._ops
from torch_npu._inductor.lowering_common import (
    add_overload,
    enable_full_lowering_fallback as enable_full_lowering_fallback_common,
    resolve_op_from_name,
)
from .. import config
from .utils import logger
from ..npu.utils import  get_anir_mode
from torch_npu._inductor.lowering_common import run_once


aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims

@run_once
def _register_npu_inductor_fallbacks():
    gen_set = set()
    fallback_set = set()
    fallback_set_exclude = set()
    env_fallback_list = config.enable_full_lowering_fallback

    if env_fallback_list:
        for op_name in env_fallback_list.split(','):
            op_name = op_name.strip()
            op = resolve_op_from_name(op_name, logger)
            if isinstance(op, torch._ops.OpOverloadPacket):
                fallback_set.add(op)
                fallback_set_exclude.add(op)
                logger.info(f"[npu|inductor|lowering|fallback] User specified fallback: {op_name}")
            else:
                logger.warning(f"[npu|inductor|lowering|fallback] Cannot resolve operator: {op_name}")

    add_overload(config.GENERATE_LIST, gen_set)

    add_overload(config.FALLBACK_LIST, fallback_set)
    
    def fallback_except_gen_set(gen_set):
        for op in lowering.lowerings:
            if op not in decomposition.decompositions and op not in gen_set:
                if isinstance(op, torch._ops.OpOverloadPacket) or \
                    isinstance(op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
                    make_fallback(op)

    def fallback_via_fallback_set(fallback_set):
        for op in lowering.lowerings:
            if op not in decomposition.decompositions and op in fallback_set:
                if isinstance(op, torch._ops.OpOverloadPacket) or \
                    isinstance(op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
                    make_fallback(op)
    
    if config.fallback_to_aten_mode not in {"off", "include", "exclude"}:
        raise AssertionError(f"Error! Unsupported fallback_to_aten_mode: {config.fallback_to_aten_mode} was set!")
    
    if get_anir_mode() == 'O0':
        fallback_except_gen_set(gen_set=[])
        decomposition.decompositions.clear()
        return 
    
    if config.fallback_to_aten_mode == 'include':
        fallback_via_fallback_set(fallback_set=fallback_set)
    elif config.fallback_to_aten_mode == 'exclude':
        fallback_except_gen_set(gen_set=gen_set)
        fallback_via_fallback_set(fallback_set=fallback_set_exclude)
    elif config.fallback_to_aten_mode == 'all':
        enable_full_lowering_fallback_common(
            lowerings,
            decomposition.decompositions,
            make_fallback,
        )
