# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Ops the triton_experimental backend must fall back to natively, explicitly.

The bulk fallback-registration pass in ``lowering.py`` walks the upstream
``lowerings`` table, so it cannot discover ops that have no upstream lowering;
they must be named here or a compile with ``implicit_fallbacks=False`` fails
with ``MissingOperatorWithoutDecomp``.  Group 1: aten ops whose decomposition
triton_experimental removes; group 2: aten ops with neither an upstream lowering
nor a decomposition at all (CUDA only compiles them through the implicit
fallback); group 3: torch_npu-private ``npu.*`` ops.  Keep one line per op,
naming the concrete ``OpOverload``: the registrar records each entry in
``FALLBACK_LIST`` as written, keeping registration and recording 1:1.
"""

import torch

from . import device_props

aten = torch.ops.aten
npu = torch.ops.npu

EXPLICIT_FALLBACK_LIST = [
    # Group 1: no upstream lowering; decomposition removed so the native kernel runs.
    aten.reflection_pad2d.default,
    aten.reflection_pad2d_backward.default,
    aten.upsample_bilinear2d.default,
    aten.upsample_bilinear2d_backward.default,
    # upsample_nearest{1,3}d_backward: disable_implicit_decomposition pops the forward
    # nearest py_kernels, so the grad graph carries the bare backward op. nearest2d has
    # an upstream lowering; the 1d/3d variants do not -- fall back natively.
    aten.upsample_nearest1d_backward.default,
    aten.upsample_nearest3d_backward.default,
    # #4937: index_fill's compiled backward reaches aten._unique.default, which has no
    # upstream lowering and no decomposition -- the bulk pass walks `lowerings`, so it
    # cannot discover the op and a strict-mode compile raises MissingOperatorWithoutDecomp.
    # Register each overload explicitly (multi-output: values/indices/inverse), and keep
    # the _unique2 family lined up so a chain-exposed sibling needs no follow-up PR.
    aten._unique.default,
    aten._unique.out,
    aten._unique2.default,
    aten._unique2.out,
    # Group 2 (#4936): aten ops with neither an upstream lowering nor a
    # decomposition -- CUDA compiles them only through the implicit fallback.
    # The .default pair is what the grad-enabled opinfo graph hits; the .out
    # pair is registered for pre-dispatch/export graphs that spell the op in
    # out= form, matching how the default backend's list pairs its
    # grid_sampler_2d_backward entries.
    aten.grid_sampler_3d.default,
    aten.grid_sampler_3d.out,
    aten.grid_sampler_3d_backward.default,
    aten.grid_sampler_3d_backward.out,
    # histogram/histogramdd: upstream makes only the bin-count overloads
    # (histogram.bin_ct, _histogramdd_bin_edges, _histogramdd_from_bin_cts)
    # fallbacks; the bins-as-tensor overloads the opinfo samples use have
    # neither a lowering nor a decomposition.
    aten.histogram.bins_tensor,
    aten._histogramdd_from_bin_tensors.default,
    # pad(mode="replicate") on a 5D input: upstream decomposes the forward and
    # makes replication_pad1d/2d_backward fallbacks, but leaves the 3d backward
    # with neither.
    aten.replication_pad3d_backward.default,
    # Group 3: torch_npu-private op, a bare node in the gather-family backward graphs;
    # non-contiguous access -- triton codegen must not lower it, native fallback only.
    npu.npu_gather_backward.default,
    # #4935: dropout private ops with no TE lowering. The native_dropout
    # decomposition routes the forward through npu._npu_dropout (packed uint8 mask);
    # once it falls back, the autograd chain exposes npu.npu_dropout_backward. Both
    # consume an NPUGeneratorImpl; native fallback keeps the RNG order identical to
    # eager.
    npu._npu_dropout.default,
    npu.npu_dropout_backward.default,
    # repeat_interleave's eager autograd formula produces this private backward op,
    # so no upstream lowering or decomposition can ever cover it.
    npu.repeat_interleave_backward_int.default,
    # KLDiv's private backward ops are decomposed instead (decomposition.py,
    # _override_kl_div_backward_decomp); also listing them here breaks backend init.
]

# A5 keeps the upstream decomposition; only other devices need a native fallback.
if not device_props.is_a5():
    EXPLICIT_FALLBACK_LIST.append(aten.embedding_dense_backward.default)
