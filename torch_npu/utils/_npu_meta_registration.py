import os
import sys
import operator
from typing import Optional
from functools import wraps, reduce, lru_cache
from typing import Callable
import torch
from torch import Tensor
from torch._ops import OpOverload, OpOverloadPacket
from torch._subclasses import fake_tensor as _subclasses_fake_tensor
from torch._C import DispatchKey
from torch._refs import div as refs_div, _broadcast_shapes
import torch._prims_common as utils
from torch._inductor import decomposition as inductor_decompo
from torch._prims_common import corresponding_real_dtype, corresponding_complex_dtype
from torch._prims_common.wrappers import out_wrapper
from torch._decomp import decomposition_table, decompositions_for_rng, get_decompositions
from torch._dynamo.symbolic_convert import break_graph_if_unsupported, InstructionTranslatorBase, stack_op
from torch._dynamo.exc import Unsupported
from torch._dynamo.variables.lists import TupleVariable
from torch._dynamo.variables.nn_module import NNModuleVariable
from torch._decomp import meta_table
import torch_npu

aten = torch.ops.aten
npu = torch.ops.npu

META_BLACKLIST = {
    "aten::empty_strided",  # causing infinite recursion, test_meta.py
    "aten::clone",  # causing infinite recursion
    "aten::_to_copy",  # causing infinite recursion, test_serialization.py -k test_tensor_subclass_getstate_overwrite  # noqa: B950
    "aten::copy_",  # Exception not raised, test_torch.py -k test_storage_meta_errors_cpu_int64  # noqa: B950
    "aten::constant_pad_nd",  # requires_grad mismatch, test_ops.py -k test_fake_crossref_backward_amp_istft_cuda_float32  # noqa: B950
    "aten::rot90",  # requires_grad mismatch! test_ops.py -k test_fake_crossref_backward_amp_rot90_cuda_float32  # noqa: B950
    "aten::as_strided_scatter",  # requires_grad mismatch, test_ops.py -k test_fake_crossref_backward_no_amp_as_strided_scatter_cuda_float32  # noqa: B950
}

INDUCTOR_DECOMP_OVERRIDE_OPS = {
    "aten::_to_copy",
}


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
        op_name = op_overload.name()
        if op_overload not in avoid_make_fallback_table:
            if op_name in INDUCTOR_DECOMP_OVERRIDE_OPS:
                inductor_decompo.decompositions[op_overload] = fn
            decomposition_table[op_overload] = fn
        op_overload.py_kernels.pop(DispatchKey.Meta, None)
        op_overload.py_impl(DispatchKey.Meta)(fn)

        _meta_library = torch.library.Library("aten", "IMPL", "Meta")
        op_name = op_overload.name()
        if op_name not in META_BLACKLIST:
            meta_table[op_overload] = fn
            key = _meta_library.ns + "/" + op_name.split("::")[-1] + "/" + _meta_library.dispatch_key
            if key in torch.library._impls:
                torch.library._impls.remove(key)
                _meta_library.impl(op_overload, fn)

    inductor_decompo.fast_random_decomps.cache_clear()
    patch_torch_decomp_decompositions()


@register_meta_npu(aten.index_put.default)
def meta_index_put_patch(self, indices, values, accumulate=False):
    return self.new_empty(self.shape)


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


@register_meta_npu(aten.native_dropout_backward)
def meta_native_dropout_backward(
    grad_output: Tensor,
    mask: Tensor,
    scale: float
):
    r = (grad_output).clone(
        memory_format=utils.suggest_memory_format(grad_output)
    )
    return r


@register_meta_npu(aten._to_copy.default)
def meta_to_copy_default(
    x,
    *,
    dtype: Optional[torch.dtype] = None,
    layout=None,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: Optional[torch.memory_format] = None,
):
    if layout and layout != torch.strided:
        raise AssertionError(f"Only strided layout is supported, got {layout}")
    if pin_memory:
        raise AssertionError("pin_memory is not supported")
    if not isinstance(x, (torch.Tensor, int, float, bool, complex)):
        raise AssertionError(f"x must be Tensor or scalar type, got {type(x)}")

    out_memory_format = memory_format if memory_format is not None else torch.contiguous_format

    if device is None and dtype is None and memory_format is None:
        if isinstance(x, torch.Tensor):
            return x.clone(memory_format=out_memory_format)
        else:
            return x
    dtype_converted = False

    if isinstance(x, torch.Tensor):
        x_tensor = x
    else:
        x_tensor = torch.scalar_tensor(x)

    if device is not None and device != x_tensor.device:
        # avoid conversions on cpu
        if dtype is not None and device.type == "cpu":
            x_tensor = torch._prims.convert_element_type(x_tensor, dtype)
            dtype_converted = True
        x_tensor = torch._prims.device_put(x_tensor, device, non_blocking)

    if dtype is not None and not dtype_converted:
        x_tensor = torch._prims.convert_element_type(x_tensor, dtype)
        dtype_converted = True

    if memory_format is not None:  # no ref/prim for memory format
        return torch.clone(x_tensor, memory_format=memory_format)
    else:
        return torch.clone(x_tensor, memory_format=out_memory_format)
