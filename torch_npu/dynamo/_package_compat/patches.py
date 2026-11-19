# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Adapters around private PyTorch package/cache interfaces.

Reuse rules follow the native CUDA/XPU compile-package contract, so most of what
this module used to do is native again:

* **A compile package is checked against the environment only.** ``SystemInfo`` is
  adapted for NPU in :mod:`.system_info`, and a package is labelled with the device
  type it was produced on. The DVM request and option identity is *not* compared
  when a package is restored, so changing an Inductor or DVM codegen knob between
  save and load reuses the saved artifacts silently, exactly as it does on GPU,
  while a chip or CANN change is still a miss. Option identity keeps its real job:
  it feeds the FX/AOT cache *keys* through
  ``torch_npu._inductor.dvm.persistence.DvmCacheIdentity``, so a
  changed knob still recompiles the graphs.
* **``caching_precompile`` is the only producer and consumer.** There is no
  ``torch.compile(package=...)`` entry, no manual package save and no load-first
  bootstrap, so ``CompilePackage.initialize`` / ``install`` and
  ``DynamoStore.save_cache_entry`` keep their native implementations.
* **Partial packages stay supported, and are handled where the automatic path
  writes them.** A bypassed code is a segment native installation skips, so it must
  not invalidate the rest of the package: only the segments that will be restored
  need to be portable, and a leftover artifact of a bypassed segment is dropped so
  it cannot make the whole write fail. An entry left with no live code is exported
  as nothing instead of as an empty hit. All of that lives in
  ``PrecompileContext.create_cache_entries``, which is what
  ``torch.compiler.save_cache_artifacts()`` calls; the per-entry save method belongs
  to the manual path and needs no adapter.
* **Nothing is re-checked while restoring.** A rejection there cannot fall back to
  a miss -- the package wrapper turns a failed reload into an error and then re-enters
  an already initialized package -- so an artifact is validated when it is written
  and only materialized when it is read.

What else stays NPU-specific is what native has no equivalent for: DVM FX fallback
code and its real constants, which native covers with Triton bundles. Those live in
``torch_npu._inductor.dvm.fx_asset``, with the rest of the DVM bodies in
:mod:`torch_npu._inductor.dvm.persistence`; the graph adapters below
(:func:`_graph_init` / :func:`_graph_restore`) only route, and every DVM import is
lazy so a cache load never loads a backend by itself.

The four adapters are split along one axis, and every branch below is one of the
two answers:

**NPU-common** -- needed by every NPU backend, decided from the record itself
(``entry.device_type``, ``has_dvm_records``). The environment record of a package
native labelled ``npu`` belongs here: ``npu`` is in ``SystemInfo.CHECK_GPUS``, so
the consumer environment-checks such a package whether or not DVM produced it.
**DVM-only** -- the DVM FX fallback protocol and package ownership, decided at
production time from the ambient request (:func:`is_dvm_request`, because a graph
that was just created carries no record yet) or from the ``_npu_*`` markers a DVM
compilation left behind. Those bodies live in
``torch_npu._inductor.dvm.persistence``.

No adapter resets user globals or suppresses an error after installation started.
Static launchers and Triton inside FX fallback assets remain out of scope.
"""
import inspect
import logging
import sys
import threading

from .context import is_dvm_request
from .format import Manifest
from .validation import has_dvm_records

log = logging.getLogger(__name__)
_INSTALL_LOCK = threading.RLock()
_installed = False  # Graph adapters, independent of the Dynamo opt-in.
_precompile_installed = False
_graph_originals = None
_package_originals = None


def _cache_entry(package):
    """Label the entry with the environment a consumer will compare."""
    from torch._dynamo import package as packages

    from . import system_info

    entry = _package_originals["cache_entry"](package)
    # Ownership comes from the artifacts recorded for this compilation, not from
    # the ambient backend choice or the entry's own device label, which is
    # exactly what is still undecided here.
    if _owns_package_entry(entry):
        from torch_npu._inductor.dvm.persistence import label_dvm_package_entry

        label_dvm_package_entry(entry, package)
    # Native takes the entry's SystemInfo from a dataclass default factory bound
    # when the class was created. On NPU that factory records neither the chip
    # name nor CANN, so writing it here is what makes the consumer's check
    # meaningful, and it is the only place it has to be written. Another
    # backend's package keeps its native metadata untouched, except this record.
    if entry.device_type == system_info.DEVICE_TYPE:
        entry.system_info = system_info.current(packages.SystemInfo)
    return entry


def _owns_package_entry(entry):
    """Whether the artifacts this entry names were produced by a DVM compile."""
    from torch._dynamo.precompile_context import PrecompileContext

    artifacts = PrecompileContext._backend_artifacts_by_key
    return has_dvm_records([artifacts[key] for key in entry.backend_ids if key in artifacts])


def _create_entries(_cls):
    """Export only what native installation could actually restore.

    ``cls`` is unused: the captured original is already the bound classmethod,
    so the adapter does not read class state.
    """
    entries, debug = _package_originals["create_entries"]()
    if any(has_dvm_records(entry) for entry in entries.values()):
        from torch_npu._inductor.dvm.persistence import portable_dvm_entries

        entries = portable_dvm_entries(entries)
    return entries, debug


def _graph_init(obj, current_callable, graph, *args, **kwargs):
    """Attach what a DVM graph needs in order to be restored elsewhere."""
    _graph_originals["graph_init"](obj, current_callable, graph, *args, **kwargs)
    if "npu" not in obj.device_types or not is_dvm_request():
        return
    # The DVM body lives with the backend it belongs to. Imported here so a
    # non-DVM compilation never loads torch_npu._inductor.dvm (C extension
    # plus SoC query) through this adapter.
    from torch_npu._inductor.dvm.persistence import attach_dvm_graph_assets

    attach_dvm_graph_assets(obj, graph)


def _graph_restore(obj, constants):
    """Restore graph-owned assets under the backend that produced them."""
    manifest = getattr(obj, "_npu_manifest", None)
    if "npu" not in obj.device_types or not isinstance(manifest, Manifest):
        # Not a DVM artifact, or one produced while no DVM identity was
        # available. Nothing NPU-specific was archived for it.
        return _graph_originals["graph_restore"](obj, constants)
    backend = manifest.request.get("backend")
    if backend is None:
        log.warning("NPU graph cache record has no backend; keeping native restore")
        return _graph_originals["graph_restore"](obj, constants)
    # Importing the DVM body loads torch_npu._inductor and therefore a backend,
    # so the scope has to cover the import, not just the call: otherwise the
    # ambient backend is loaded first and the producer's has to replace it.
    from torch_npu.utils._dynamo import _NpuBackendScope

    with _NpuBackendScope(backend):
        from torch_npu._inductor.dvm.persistence import restore_dvm_graph_assets

        return restore_dvm_graph_assets(obj, constants, _graph_originals["graph_restore"])


def _probe_package_adapters():
    """Decide whether the native Dynamo package APIs still match what we adapt.

    ``torch._dynamo.package`` / ``precompile_context`` are private APIs whose
    shape drifts between torch versions. A mismatch must degrade to a bypass
    instead of crashing the install, so the probe checks, in order: the classes
    exist, ``CompilePackage.initialize`` carries the ``ignore_inlined_sources``
    parameter that marks the package semantics we adapt, and every member the
    adapters rebind is still there. Returns the modules plus the verdict.

    All-or-nothing on purpose: installing only part of this set would leave a
    package labelled ``npu`` with a ``SystemInfo`` that records no CANN version,
    which the saving host itself rejects.
    """
    packages = PrecompileContext = None
    try:
        from torch._dynamo import package as packages
        from torch._dynamo.precompile_context import PrecompileContext

        available = all(hasattr(packages, name) for name in ("CompilePackage", "SystemInfo"))
        if available:
            available = "ignore_inlined_sources" in inspect.signature(
                packages.CompilePackage.initialize
            ).parameters
        if available:
            # The adapters also bind members the checks above do not cover;
            # probe them so a renamed private API degrades to a bypass
            # instead of crashing the install.
            available = (
                hasattr(packages.CompilePackage, "cache_entry")
                and hasattr(PrecompileContext, "create_cache_entries")
                and hasattr(PrecompileContext, "_backend_artifacts_by_key")
            )
    except (ImportError, AttributeError, TypeError, ValueError):
        log.debug("Failed to inspect native Dynamo package APIs", exc_info=True)
        available = False
    return packages, PrecompileContext, available


def _probe_graph_adapters():
    """Import and validate the native Inductor APIs the graph adapters rebind.

    Returns ``(CompiledFxGraph, available)``. Dynamo can be imported *by*
    Inductor's standalone_compile module, and OutputCode would then import a
    still-undefined AOTCompiledArtifact, so the probe first waits for Inductor to
    finish initializing and reports "not available" while it has not: the caller
    retries at the next compile entry. The import stays behind that readiness
    check for the same reason.
    """
    for name in ("torch._inductor", "torch._inductor.standalone_compile"):
        spec = getattr(sys.modules.get(name), "__spec__", None)
        if getattr(spec, "_initializing", False):
            return None, False
    from torch._inductor.output_code import CompiledFxGraph

    # Only the member we actually rebind is probed. A symbol nothing calls would
    # turn an upstream rename into this whole layer silently going uninstalled.
    available = hasattr(CompiledFxGraph, "after_deserialization")
    if not available:
        log.info("NPU graph-cache adapters unavailable on this PyTorch version")
    return CompiledFxGraph, available


def install():
    """Install graph adapters, plus package adapters while caching_precompile is on."""
    global _installed, _precompile_installed, _graph_originals, _package_originals
    with _INSTALL_LOCK:
        from torch._dynamo import config as dynamo_config

        requested = getattr(dynamo_config, "caching_precompile", False)
        if _installed and (not requested or _precompile_installed):
            return True

        packages, PrecompileContext, available = _probe_package_adapters()
        if requested and not available:
            log.info("NPU Dynamo-package adapters unavailable; graph-cache adapters remain enabled")

        if requested and available and not _precompile_installed:
            from . import system_info

            # Environment-only reuse needs the NPU values native SystemInfo does not
            # collect. Installed before any entry can be created or compared.
            system_info.install(packages)
            if _package_originals is None:
                _package_originals = {"cache_entry": packages.CompilePackage.cache_entry,
                                      "create_entries": PrecompileContext.create_cache_entries}
            packages.CompilePackage.cache_entry = _cache_entry
            PrecompileContext.create_cache_entries = classmethod(_create_entries)
            _precompile_installed = True

        CompiledFxGraph, graph_available = _probe_graph_adapters()
        if not graph_available:
            return False

        if not _installed:
            if _graph_originals is None:
                _graph_originals = {"graph_init": CompiledFxGraph.__init__,
                                    "graph_restore": CompiledFxGraph.after_deserialization}
            CompiledFxGraph.__init__ = _graph_init
            CompiledFxGraph.after_deserialization = _graph_restore
            _installed = True
        return _installed and (not requested or _precompile_installed)
