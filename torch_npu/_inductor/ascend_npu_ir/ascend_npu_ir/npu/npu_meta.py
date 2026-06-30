from functools import lru_cache
from typing import Callable
import torch
from torch._ops import OpOverload, OpOverloadPacket
from torch._subclasses import fake_tensor as _subclasses_fake_tensor
from torch._C import DispatchKey
from torch._decomp import decomposition_table
from torch_npu._inductor.lowering_common import run_once

aten = torch.ops.aten
npu = torch.ops.npu

npu_meta_table = {}
break_fn_table = {}
break_mapping_table = {}
avoid_make_fallback_table = []


def _add_op_to_meta_table(op, fn, avoid_fallback_flag=False):
    overloads = []
    if isinstance(op, OpOverload):
        overloads.append(op)
    else:
        if not isinstance(op, OpOverloadPacket):
            raise AssertionError("op must be instance of OpOverloadPacket.")
        for ol in op.overloads():
            overloads.append(getattr(op, ol))

    for op_overload in overloads:
        if op_overload in npu_meta_table:
            raise RuntimeError(f"duplicate registrations for npu_meta_table {op_overload}")
        npu_meta_table[op_overload] = fn
        if avoid_fallback_flag:
            avoid_make_fallback_table.append(op_overload)

def patch_torch_decomp_decompositions():
    '''
    Because source torch_decomp_decompositions only enable the decompositions in
    torch/_decomp/decompositions.py. Patch it to make decompositions in this file work.
    '''
    src_func = _subclasses_fake_tensor.torch_decomp_decompositions

    @lru_cache(None)
    def torch_decomp_decompositions_new(func):
        if func in npu_meta_table.keys():
            return True
        return src_func(func)
    _subclasses_fake_tensor.torch_decomp_decompositions = torch_decomp_decompositions_new

def register_meta_npu(op, avoid_fallback_flag=False):
    def meta_decorator(fn: Callable):
        _add_op_to_meta_table(op, fn, avoid_fallback_flag)
        return fn

    return meta_decorator

@run_once
def npu_patch_meta():
    '''
    Torch official register decompostions and meta func for some aten ops,
    which will raise conflict when npu outputs' dtype and shape are different
    from native impl. Delete decompositions and meta func of these ops and add
    npu decompositions and meta func.
    '''
    for op_overload, fn in npu_meta_table.items():
        if not isinstance(op_overload, OpOverload):
            raise AssertionError("op_overload must be instance of OpOverload.")
        if op_overload not in avoid_make_fallback_table:
            decomposition_table[op_overload] = fn
        op_overload.py_kernels.pop(DispatchKey.Meta, None)
        op_overload.py_impl(DispatchKey.Meta)(fn)

    patch_torch_decomp_decompositions()
