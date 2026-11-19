# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Adapters that let a DVM compilation survive a process boundary.

``torch._dynamo.config.caching_precompile`` gives every ``torch.compile`` call a
``CompilePackage``: Dynamo code, guards, resume state and the compiler's backend
artifacts, saved and loaded as one unit. On CUDA/XPU the package is reused
whenever the environment matches, and nothing else about the previous compilation
is re-examined. On Ascend the same contract holds, and this package keeps only
what native has no equivalent for, split by ownership:

NPU-common (stays here, safe for every backend and every CPU-only host):

1. **Environment identity for NPU.** ``SystemInfo`` only knows CUDA and XPU, so on
   Ascend it recorded no chip and no CANN version and every package looked
   compatible with every other. :mod:`.system_info` fills those native fields in,
   which makes a chip or CANN change a miss and nothing else.
2. **The artifact record format.** :mod:`.format` owns canonical serialization,
   hashing and the ``Manifest`` envelope, shared by every NPU artifact.
3. **The backend-neutral compile scope.** :mod:`.context` owns the context
   variables, the per-backend scope, device discovery and the ``is_dvm_request``
   predicate that the adapters route on.
4. **Ownership routing.** :mod:`.validation` tells whose record a restored object
   is, without reading any backend's configuration.

DVM-only (``torch_npu._inductor.dvm``, imported lazily so a non-DVM compilation
or a CPU-only host never loads the DVM extension):

* the producer identity -- which DVM/ANIR knobs a compilation is keyed on and
  the compiler-installation fingerprint (``dvm.persistence``);
* the request context and the scope that freezes it;
* the graph asset attach/restore bodies and DVM record validation;
* package entry labelling/trimming;
* the ``DvmCacheIdentity`` pass that publishes the identity into FX/AOT keys;
* the FX asset archive (``dvm.fx_asset``). Both its producer and every reader of
  its ``npu_fx_asset_id`` belong to DVM; the ``ascend_npu_ir`` read is a lazy
  branch that only fires for a key DVM wrote.

What is deliberately *not* here, compared with an earlier implementation:

* No manual entry (``torch.compile(..., package=...)``), no manual package save, no
  load-first bootstrap. The automatic ``caching_precompile`` path is the only
  producer and consumer, so ``CompilePackage.initialize`` / ``install`` and the
  per-entry save method keep their native implementations.
* No option or request comparison when a package loads. A changed Inductor/DVM knob
  reuses the saved package silently, exactly as on GPU; it still changes the FX/AOT
  keys, so graphs compiled under other options are recompiled.
* No validation while restoring. Rejecting there cannot fall back to a miss, so an
  artifact is checked when it is written and only materialized when it is read.

Keep every module in this package independent of ``torch_npu._inductor`` at import
time: loading a cache does not choose a target or a backend. Other NPU backends
keep their native compilation and cache protocols and are not marked uncacheable
from here.
"""


def install():
    """Install graph adapters, plus package adapters while caching_precompile is on."""
    from .patches import install as install_patches

    return install_patches()
