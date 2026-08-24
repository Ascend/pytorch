# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Op registration lists for the triton_experimental backend.

``GENERATE_LIST``: ops the backend keeps an inductor lowering for (a kernel is
generated). Every other op that is not a decomposition is turned into a fallback by
``lowering._register_npu_inductor_fallbacks``. ``KEEP_UPSTREAM_LOWERING``: ops whose
upstream lowering must be preserved verbatim rather than clobbered into a fallback.
"""
import torch
from torch._inductor.fx_passes.control_dependencies import control_deps

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
    aten.exp2,
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
    aten.tan,
    aten.reciprocal,
    aten.relu,
    aten.where,
    aten.log,
    aten.log2,
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
    aten.glu
]
# A5 (910_95) ONLY: the CANN indirect-mem extension ops behind these exist only
# on A5 (fail to lower on A2/A3/910B). Mirrors torch_npu's INDIRECT_MEM_OVERRIDE_LIST.
if _device_props.is_a5():
    GENERATE_LIST += []

# Higher-order and runtime-assertion ops whose intentional upstream lowerings must
# not be replaced with generic fallbacks.  control_deps carries a Subgraph argument
# that only its dedicated lowering understands; FallbackKernel treats it as a tensor
# and attempts to read a nonexistent dtype.  For aten._assert_* ops, a fallback
# re-invokes the real aten op during lowering and can throw after guards are folded
# to Python bools (for example, aten._assert_scalar(True, msg)).
KEEP_UPSTREAM_LOWERING = [control_deps] + [
    getattr(aten, _name)
    for _name in (
        "_assert_scalar",
        "_assert_tensor_metadata",
        "_assert_async",
        "_functional_assert_async",
        # index_put / index_put_: a custom lowering (npu_index_put / npu_index_put_
        # in lowering.py) casts values to self.dtype then routes to the extern
        # ir.IndexPutFallback (aten.index_put_) instead of the upstream ir.Scatter
        # kernel. Keep them here so _register_npu_inductor_fallbacks does NOT clobber
        # that custom lowering back into a bare make_fallback -- without the cast, fp16
        # self / fp32 values hits aclnnIndexPutImpl EZ1001 (self/values dtype mismatch).
        "index_put",
        "index_put_",
    )
    if hasattr(aten, _name)
]

# _inductor_test.realize (test_operators.realize) is a fusion-barrier op: upstream
# registers an IR-level lowering (_realize: x.realize(); return clone(x)) where
# x.realize() forces the lazy Pointwise/Reduction node into a ComputedBuffer,
# physically preventing the scheduler from fusing across it. make_fallback would
# instead register an ir.FallbackKernel -- the eager op falls back off-device and
# the FallbackKernel itself becomes an extra scheduler node, so a graph that cuda
# segments into 3 nodes (e.g. +1 | realize | *2 | realize) yields 6 here, breaking
# the len(nodes)==3 assertion in test_inner_fn_str_and_stride_npu. Keep upstream's
# IR-level lowering (same mechanism the ascend_npu_ir backend already uses) so the
# realize barrier segments the graph correctly.
# The attribute access above must not crash backend activation: _inductor_test::
# realize is registered by torch._inductor.test_operators, which torch._dynamo.
# trace_rules imports unconditionally (torch >= 2.3.0), so the op is always
# registered once a compile -- and therefore this backend -- activates; upstream
# relies on the same chain (torch/_inductor/lowering.py registers _realize at
# module import).  Resolve defensively so activation never depends on that
# incidental import order: if the op is not registered it cannot appear in any
# graph, so there is nothing to keep.
_realize = getattr(torch.ops._inductor_test, "realize", None)
if _realize is not None:
    KEEP_UPSTREAM_LOWERING.append(_realize)
