import os
import sys
import operator
from functools import wraps, reduce, lru_cache
from typing import Callable, Optional
import torch
from torch import Tensor
from torch._ops import OpOverload, OpOverloadPacket
from torch._subclasses import fake_tensor as _subclasses_fake_tensor
from torch._C import DispatchKey
from torch._refs import div as refs_div, _broadcast_shapes
from torch._inductor import decomposition as inductor_decompo
from torch._prims_common import corresponding_real_dtype, corresponding_complex_dtype
from torch._prims_common.wrappers import out_wrapper
from torch._decomp import decomposition_table, decompositions_for_rng, get_decompositions
from torch._dynamo.symbolic_convert import break_graph_if_unsupported, InstructionTranslatorBase, stack_op
from torch._dynamo.exc import Unsupported
from torch._dynamo.variables.lists import TupleVariable
from torch._dynamo.variables.nn_module import NNModuleVariable
import torch_npu

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
avoid_make_fallback_table = []
inductor_decomp_table = []


def _add_op_to_meta_table(op, fn, avoid_fallback_flag=False, inductor_decomp=False):
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
        if inductor_decomp:
            inductor_decomp_table.append(op_overload)


def patch_torch_inductor_decompositions():
    '''
    TorchInductor traces compiled backward with its own decomposition table.
    Only patch ops that explicitly opted in via inductor_decomp=True so we
    don't accidentally overwrite unrelated inductor decompositions.
    '''
    import torch._inductor.decomposition as inductor_decomposition

    for op_overload in inductor_decomp_table:
        if op_overload in npu_meta_table:
            inductor_decomposition.decompositions[op_overload] = npu_meta_table[op_overload]


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


def register_meta_npu(op, avoid_fallback_flag=False, inductor_decomp=False):
    def meta_decorator(fn: Callable):
        _add_op_to_meta_table(op, fn, avoid_fallback_flag, inductor_decomp)
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

    inductor_decompo.fast_random_decomps.cache_clear()
    patch_torch_decomp_decompositions()
    patch_torch_inductor_decompositions()



@register_meta_npu(aten.index_put.default)
def meta_index_put_patch(self, indices, values, accumulate=False):
    return self.new_empty(self.shape)


@register_meta_npu(aten.native_dropout, inductor_decomp=True)
@out_wrapper("out0", "out1")
def meta_native_dropout_patch(tensor_input: Tensor, p: float, train: Optional[bool]):
    if torch._inductor.config.fallback_random:
        if train and p != 0:
            if tensor_input.is_meta:
                numel = reduce(operator.mul, tensor_input.shape)
                numel = (numel + 128 - 1) // 128 * 128
                numel = numel // 8
                return (
                    torch.empty_like(tensor_input),
                    torch.empty(numel, dtype=torch.uint8, device=tensor_input.device),
                )
            return torch.ops.npu._npu_dropout(tensor_input, p)
        return (tensor_input, torch.ones_like(tensor_input, dtype=torch.bool))
    else:
        from torch._decomp.decompositions import native_dropout
        return native_dropout(tensor_input, p, train)


@register_meta_npu(aten.native_dropout_backward, inductor_decomp=True)
@out_wrapper()
def meta_native_dropout_backward_patch(grad_output: Tensor, mask: Tensor, scale: float):
    if torch._inductor.config.fallback_random:
        if grad_output.is_meta:
            return torch.empty_like(grad_output)
        p = 1 if scale == 0 else (1 - 1 / scale)
        return torch.ops.npu.npu_dropout_backward(grad_output, mask, p)
    else:
        from torch._decomp.decompositions import native_dropout_backward
        return native_dropout_backward(grad_output, mask, scale)