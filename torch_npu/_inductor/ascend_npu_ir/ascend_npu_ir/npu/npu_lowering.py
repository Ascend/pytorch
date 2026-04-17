import os
import sys
import time
from functools import reduce

from torch._inductor import lowering
from torch._inductor.lowering import lowerings, make_fallback
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
    
    def enable_full_lowering_fallback():
        ops_to_fallback = list(filter(
            lambda op: op not in decomposition.decompositions and
                isinstance(op, (torch._ops.OpOverloadPacket, torch._ops.OpOverload, torch._ops.HigherOrderOperator)),
            lowerings
        ))
        for op in ops_to_fallback:
            make_fallback(op)

        _fallback_ops_with_meta()
    
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
    elif config.fallback_to_aten_mode == 'all':
        enable_full_lowering_fallback()


def get_nested_attr(obj, attr_path, default=None):
    try:
        return reduce(getattr, attr_path.split('.'), obj)
    except AttributeError:
        return default
    

def _fallback_ops_with_meta():
    """
    Fallback all ops that have a Meta implementation but are not yet in lowerings
    """
    all_ops = torch._C._dispatch_get_all_op_names()

    for op_name in all_ops:
        has_meta = torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "Meta")
        has_comp = torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "CompositeImplicitAutograd")

        if not (has_meta or has_comp):
            continue

        try:
            namespace, name_with_overload = op_name.split(".", 1)
        except ValueError:
            continue

        if "." in name_with_overload:
            name, overload = name_with_overload.split(".", 1)
        else:
            name, overload = name_with_overload, "default"

        normalized_path = f"{namespace}.{name}.{overload}"
        op_overload = get_nested_attr(torch.ops, normalized_path)
        if not isinstance(op_overload, torch._ops.OpOverload):
            continue

        if op_overload in lowerings or op_overload in decomposition.decompositions:
            continue

        make_fallback(op_overload)
