import os
import sys
import operator
from functools import wraps, reduce, lru_cache
from typing import Callable, Optional, Tuple
import torch
import torch_npu
from torch import Tensor
from torch._ops import OpOverload, OpOverloadPacket
from torch._subclasses import fake_tensor as _subclasses_fake_tensor
from torch._C import DispatchKey
from torch._refs import div as refs_div, _broadcast_shapes
from torch._prims_common import corresponding_real_dtype, corresponding_complex_dtype
from torch._prims_common.wrappers import out_wrapper
from torch._decomp import decomposition_table, decompositions_for_rng, get_decompositions
from torch._dynamo.symbolic_convert import break_graph_if_unsupported, InstructionTranslatorBase, stack_op
from torch._dynamo.exc import Unsupported
from torch._dynamo.variables.lists import TupleVariable
from torch._dynamo.variables.nn_module import NNModuleVariable


aten = torch.ops.aten
npu = torch.ops.npu


def run_once(f):
    """Runs a function (successfully) only once.
    The running can be reset by setting the `has_run` attribute to False
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            result = f(*args, **kwargs)
            wrapper.has_run = True
            return result
        return None
    wrapper.has_run = False
    return wrapper


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


@register_meta_npu(aten.native_dropout)
def meta_native_dropout(tensor_input: Tensor, p: float, train: Optional[bool]):
    if train and p != 0:
        sizes_1 = tensor_input.shape
        numel = reduce(operator.mul, sizes_1)
        numel = (numel + 128 - 1) // 128 * 128
        numel = numel // 8
        return (torch.empty_like(tensor_input), torch.empty(numel, dtype=torch.uint8, device=tensor_input.device))
    else:
        return (tensor_input, torch.ones_like(tensor_input, dtype=torch.bool))

@register_meta_npu(npu.npu_fusion_attention)
def npu_fusion_attention_forward(query, key, value, head_num, input_layout, pse=None, padding_mask=None,
                                atten_mask=None, scale=1.0, keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647,
                                inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False):
    B = query.size(0)
    N = head_num
    S1 = query.size(2)
    S2 = key.size(2)

    if input_layout == "BSH" or input_layout == "BSND":
        B = query.size(0)
        S1 = query.size(1)
        S2 = key.size(1)

    if input_layout == "SBH":
        B = query.size(1)
        S1 = query.size(0)
        S2 = key.size(0)

    seed = 0
    offset = 0
    numels = 0
    return (torch.empty_like(query).contiguous(),
            query.new_empty([B, head_num, S1, 8], dtype=torch.float32),
            query.new_empty([B, head_num, S1, 8], dtype=torch.float32),
            query.new_empty([0]),
            seed,
            offset,
            numels)

@register_meta_npu(npu.npu_fusion_attention_grad)
def npu_fusion_attention_backward(query, key, value, dy, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None,
                                  softmax_max=None, softmax_sum=None, softmax_in=None, attention_in=None, scale_value=1.0,
                                  keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, seed=0, offset=0,
                                  numels=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False):
    return (torch.empty_like(query).contiguous(), torch.empty_like(key).contiguous(), torch.empty_like(value).contiguous(), query.new_empty([0]))
