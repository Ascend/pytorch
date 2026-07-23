# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""Backend-activation orchestrator for the triton_experimental NPU backend.

``apply_npu_overrides`` is the single entry point called by ``__init__._activate``. It
sequences every NPU override, but the overrides themselves now live next to the
layer they patch:

* decomposition / dispatcher / SDPA overrides -> ``.._inductor.decomposition``
  (``_register_triton_experimental_decompositions``, called directly by the
  loader ``_load_triton_experimental_backend`` -- NOT from here)
* FX graph passes (loop-merge fold, int->float->int elision, addmm fusion) ->
  ``fx_passes.py``
* Triton codegen / scheduler patches (block hints, 0-dim CPU, int64->int32,
  check_config) -> ``codegen/triton.py``

What remains here is what doesn't belong to any single layer: plain inductor /
dynamo ``config`` flag flips and the pointwise autotuning gate that rebinds our
own heuristics module. Importing this module still has the config side effect of
turning ``size_asserts`` off.
"""
import logging

from . import config as ncfg
from .fx_passes import (
    _disable_addmm_fusion_pass,
    _install_elide_int_float_int_pass,
    _install_fold_max1_in_loop_merge,
)
from .codegen.triton import apply_npu_codegen_patches

import torch._inductor.config as inductor_config
inductor_config.size_asserts = False

log = logging.getLogger("torch._inductor")


# disable_pointwise_autotuning gates all three NPU autotune entry points. It
# returns True whenever torch.use_deterministic_algorithms is on (set by
# torchbench), silently locking every kernel to its default tile — far slower on
# NPU (e.g. a 6.3M-elem gelu_backward runs ~154 inner iters/core). Autotune here
# is timing-only (numerics unaffected), so ignore the deterministic flag. 2.13
# inlined the upstream helper; npu_triton_heuristics reintroduces a local gate —
# rebind that module attribute so all three entry points see the NPU version.
def _override_disable_pointwise_autotuning():
    def _npu_disable_pointwise_autotuning(inductor_meta):
        return not inductor_meta.get("autotune_pointwise", True)

    import sys
    npu_th = sys.modules.get("torch_npu._inductor.triton_experimental.npu_triton_heuristics")
    if npu_th is not None:
        npu_th.disable_pointwise_autotuning = _npu_disable_pointwise_autotuning


# torch._dynamo use_recursive_dict_tags_for_guards (new in 2.13, default True) is
# a guard fast-path that recursively checks nested-module __dict__ version tags.
# On speech_transformer (deep encoder/decoder ModuleLists, --train) the recursive
# C++ guard walk does THPVariable_Check on a borrowed None and over-decrefs the
# Py_None singleton -> none_dealloc -> abort("deallocating None") the first time
# the compiled backward's guards run. Disabling drops to the plain recursive check
# (identical guards, only slightly slower) — no correctness cost.
def _disable_recursive_dict_tag_guards():
    if not ncfg.disable_recursive_dict_tag_guards:
        return
    try:
        from torch._dynamo import config as _dynamo_config
    except Exception:
        return
    if hasattr(_dynamo_config, "use_recursive_dict_tags_for_guards"):
        _dynamo_config.use_recursive_dict_tags_for_guards = False


def _disable_pad_mm_pass():
    # Disable ONLY the shape-padding mm/addmm/bmm pass (pad_mm.py). It pads M/N/K
    # up to an alignment and rewrites addmm into padded-matmul + slice, which the
    # NPU backend can't handle; every other FX pass (incl add+mm->addmm) stays on.
    # config.shape_padding is the first conjunct of should_pad_common(), so setting
    # it False short-circuits the whole pad family and nothing else.
    if not ncfg.disable_pad_mm:
        return
    from torch._inductor import config as inductor_config

    inductor_config.shape_padding = False


def apply_npu_overrides():
    # config-flag flips + FX passes that don't touch the decomposition table.
    _override_disable_pointwise_autotuning()
    _disable_recursive_dict_tag_guards()
    _disable_pad_mm_pass()
    _disable_addmm_fusion_pass()
    _install_fold_max1_in_loop_merge()
    _install_elide_int_float_int_pass()

    # NOTE: decomposition-table + dispatcher + SDPA overrides are no longer applied
    # here. They live in torch_npu._inductor.decomposition
    # (_register_triton_experimental_decompositions) and are invoked directly by the
    # loader _load_triton_experimental_backend before _activate(), mirroring the
    # other backends' _register_*_decompositions entry points.

    # Triton codegen / scheduler monkeypatches (codegen/triton.py).
    apply_npu_codegen_patches()
