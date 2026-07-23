# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Op registration lists for the triton_experimental backend.

``GENERATE_LIST``: ops the backend keeps an inductor lowering for (a kernel is
generated). Every other op that is not a decomposition is turned into a fallback by
``lowering._register_npu_inductor_fallbacks``. ``KEEP_UPSTREAM_LOWERING``: ops whose
upstream lowering must be preserved verbatim rather than clobbered into a fallback.
"""
import torch

from . import device_props as _device_props

aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims
npu = torch.ops.npu

GENERATE_LIST = [
    # Pointwise ops
    aten.mul,
    aten.add,
    aten.sub,
    aten.div,
    aten.exp,
    aten.pow,
    aten.rsqrt,
    aten.neg,
    aten.lt,
    aten.gt,
    aten.ge,
    aten.le,
    aten.eq,
    aten.sigmoid,
    prims.convert_element_type,
    npu.npu_dtype_cast,
    npu.npu_dtype_cast_backward,
    npu._npu_dtype_cast,
    npu._npu_dtype_cast_backward,
    aten.sin,
    aten.cos,
    aten.reciprocal,
    aten.relu,
    aten.where,
    aten.log,
    aten.sqrt,
    aten.clamp_min,
    aten.clamp_max,
    aten.bitwise_not,
    aten.tanh,
    aten.copy,
    aten.copy_,

    # Non-pointwise ops
    aten.squeeze,
    aten.unsqueeze,
    aten.expand,
    aten.repeat,
    aten.clone,
    aten.reshape,
    aten.var_mean,
    aten.sum,
    aten.mean,
    aten.full,
    aten.slice,
    aten.select,
    aten.split,
    aten.permute,
    aten.amax,
    aten.cat,
    aten.slice_scatter,
    aten.scalar_tensor,
    aten.unbind,
    aten.lift_fresh_copy,
    aten.var,
    aten.erf,
    prims.device_put,
    aten.abs,
    aten.max,
    aten.amin,
    aten.slice_scatter,
    aten.select_scatter,
    npu._npu_dropout,
    aten.empty,
    aten.copy_,
    aten.split_with_sizes,
    aten.ne,
    aten.bitwise_or,
    aten.bitwise_and,
    aten.minimum,
    aten.maximum,
    prims.iota,
    aten.logical_not,
    aten.mm,
    aten.convolution,
    aten.convolution_backward,
    aten.bmm,
    aten.addmm,
]

# A5 (910_95) ONLY: the CANN indirect-mem extension ops behind these exist only
# on A5 (fail to lower on A2/A3/910B). Mirrors torch_npu's INDIRECT_MEM_OVERRIDE_LIST.
if _device_props.is_a5():
    GENERATE_LIST += []

# Runtime-assertion ops (torch._check / aten._assert_*): upstream registers
# intentional lowerings for these -- _assert_scalar / _assert_tensor_metadata are
# no-ops (the checks are emitted at codegen time via deferred runtime asserts),
# while _assert_async / _functional_assert_async emit an in-kernel device_assert.
# They must NOT be converted to fallbacks: a fallback re-invokes the real aten op
# during lowering, which throws when the guard has been constant-folded to a
# python bool, e.g. torch._check(s0 == s0) -> aten._assert_scalar(True, msg):
#   RuntimeError: aten::_assert_scalar() ... Cannot cast True to number
# Keep whatever upstream registered for these ops instead of clobbering it.
KEEP_UPSTREAM_LOWERING = [
    getattr(aten, _name)
    for _name in (
        "_assert_scalar",
        "_assert_tensor_metadata",
        "_assert_async",
        "_functional_assert_async",
    )
    if hasattr(aten, _name)
]
