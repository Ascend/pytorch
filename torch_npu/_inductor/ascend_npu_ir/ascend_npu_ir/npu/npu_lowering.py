import os
import sys
import time
from torch._inductor import lowering
from torch._inductor.lowering import make_fallback
from torch._inductor import decomposition
import torch._ops
from .. import config
from ..npu.utils import run_once, get_anir_mode

aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims

@run_once
def _register_npu_inductor_fallbacks():
    gen_set = set()
    fallback_set = set()

    for fn in config.GENERATE_LIST:
        gen_set.add(fn)
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                gen_set.add(other_fn)

    for fn in config.FALLBACK_LIST:
        fallback_set.add(fn)
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                fallback_set.add(other_fn)
    
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