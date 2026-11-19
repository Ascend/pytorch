# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""One DVM compilation's producer request and the scope that freezes it.

The context is captured once per compilation, bound through
:func:`request_scope`, and published into Inductor's backend pass UUID by
:mod:`.cache_identity` so that the environment and the effective codegen
options reach the FX/AOT cache keys.

This is deliberately **not** a load-time compatibility contract. A
``CompilePackage`` is reused whenever the native ``SystemInfo`` accepts the
environment, exactly as on CUDA/XPU (see ``_package_compat.system_info``).
Options that differ between save and load are no reason to reject a package;
they only change the lower cache keys, so a graph compiled under other options
is a lower-cache miss and gets recompiled.
"""
import contextlib
import dataclasses
import functools
import importlib
import logging
import sys

from torch_npu.dynamo._package_compat.context import ACTIVE
from torch_npu.dynamo._package_compat.format import (
    IdentityUnavailable, Manifest, canonical,
)

from .identity import (
    INITIALIZATION_FIELDS, _cache_snapshot, _options_identity,
    capture_configs, collect_identity, initialization_state,
)

log = logging.getLogger(__name__)

SHAPE_FIELDS = ("enable_shape_handling", "shape_handling_configs", "shape_handling_dict")


@dataclasses.dataclass
class CompileRequest:
    """Effective compiler options; cache snapshots never replace live callbacks."""
    options: dict
    identity_options: dict | None = dataclasses.field(default=None, init=False, repr=False)
    reason: str | None = dataclasses.field(default=None, init=False)

    def __post_init__(self):
        self.options = dict(self.options)
        try:
            self.identity_options = _options_identity(self.options)
        except IdentityUnavailable as error:
            self.reason = str(error)

    def check_identity(self):
        """Guard one long-lived request against in-place edits of its own options."""
        if _options_identity(self.options) != self.identity_options:
            raise IdentityUnavailable("NPU_CONFIG_IDENTITY_MISMATCH:compiler_options")


@dataclasses.dataclass
class RequestContext:
    """The producer request one compilation is keyed by."""
    manifest: Manifest | None
    backend: str
    devices: tuple
    configs: dict
    persistable: bool = True
    reason: str | None = None
    request: CompileRequest | None = None

    @classmethod
    def uncacheable(cls, backend, devices, reason="NPU_REQUEST_UNCACHEABLE", *, request=None):
        """Bind a bypass to FX/AOT and deferred compilation without a snapshot."""
        return cls(None, backend, devices, {}, False, reason, request)


def _compile_option_value(options, field):
    """Read one npu compile option following the active option spelling.

    torch with pytorch#194998 namespaces the keys ("npu.<field>") in the
    compile options and owns their globals in
    torch_npu._inductor.compile_options; older torch keeps the bare
    spellings injected into torch._inductor.config. An explicit entry in
    ``options`` (either spelling) wins over the owning module's current
    value, mirroring the pre-refactor behaviour.
    """
    from torch_npu._compat.inductor import (
        compat_compile_option, namespaced_compile_options,
    )

    namespaced = compat_compile_option(field)
    if namespaced in options or field in options:
        return options.get(namespaced, options.get(field))
    if namespaced_compile_options():
        module = sys.modules.get("torch_npu._inductor.compile_options")
        return getattr(module, field, None)
    import torch._inductor.config as inductor_config

    return getattr(inductor_config, field, None)


def make_context(devices, request):
    """Capture the producer identity of one DVM compilation."""
    import torch
    from torch_npu.utils._dynamo import _resolve_npu_backend

    backend = _resolve_npu_backend(
        _compile_option_value(request.options, "npu_backend")
    )
    context = RequestContext.uncacheable(backend, devices, request.reason, request=request)
    if request.reason is not None:
        return context
    if not devices or any(device.index is None for device in devices):
        context.reason = "NPU_TARGET_UNKNOWN"
        return context
    try:
        request.check_identity()
        configs = capture_configs(backend)
        shape = {field: _compile_option_value(request.options, field)
                 for field in SHAPE_FIELDS}
        state = initialization_state(configs)
        manifest = Manifest.create({"backend": backend, "configs": configs, "shape": shape,
                                    "initialization": state,
                                    "deterministic_level": torch.npu._get_deterministic_level(),
                                    "device_indices": [device.index for device in devices],
                                    "options": request.identity_options},
                                   collect_identity(devices, backend))
    except Exception as error:
        context.reason = (str(error) if isinstance(error, IdentityUnavailable)
                          else f"NPU_CACHE_IDENTITY_UNAVAILABLE:{type(error).__name__}:{error}")
        log.info("NPU cache identity unavailable: %s", context.reason, exc_info=True)
        return context
    return RequestContext(manifest, backend, devices, configs, request=request)


@contextlib.contextmanager
def configuration_scope(configs):
    """Freeze codegen knobs for the duration of one compile."""
    saved = []
    try:
        for module_name, values in configs.items():
            module = importlib.import_module(module_name)
            for path, value in values.items():
                if path in INITIALIZATION_FIELDS.get(module_name, ()):
                    continue
                parent = module
                parts = path.split(".")
                for name in parts[:-1]:
                    parent = getattr(parent, name)
                old = getattr(parent, parts[-1])
                if old is not value:
                    saved.append((parent, parts[-1], old))
                    setattr(parent, parts[-1], _cache_snapshot(value, f"{module_name}.{path}"))
        yield
    finally:
        for parent, name, value in reversed(saved):
            setattr(parent, name, value)


@contextlib.contextmanager
def request_scope(context):
    """Freeze one compilation's codegen knobs at its boundaries."""
    import torch

    token = ACTIVE.set(context)
    try:
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(configuration_scope(context.configs))
                if context.persistable and context.request is not None:
                    # A live callback that changed must not authorize the captured
                    # key, including when backward is compiled later.
                    context.request.check_identity()
            except IdentityUnavailable as error:
                context.manifest, context.configs = None, {}
                context.persistable, context.reason = False, str(error)
            if not context.persistable:
                # A graph whose identity is unknown must not enter a cache: a
                # later process could neither validate nor key it.
                if context.request is not None:
                    context.request.options["force_disable_caches"] = True
                stack.enter_context(torch._inductor.config.patch(force_disable_caches=True))
            yield
    finally:
        ACTIVE.reset(token)


def bind_inner_compile(inner, context):
    """Bind deferred AOT compilation to its NPU request."""
    if context is None or getattr(inner, "_npu_request", None) is context:
        return inner

    @functools.wraps(inner)
    def bound(*args, **kwargs):
        from torch._dynamo.convert_frame import compile_lock
        from torch_npu.utils._dynamo import _NpuBackendScope

        with compile_lock, _NpuBackendScope(context.backend):
            if context.persistable:
                try:
                    # Initialization-only fields are not replayed. Deferred
                    # backward must check live policy/tables before using the
                    # forward's identity for another cache lookup or save.
                    if canonical(initialization_state()) != context.manifest.request.get("initialization"):
                        raise IdentityUnavailable("NPU_INITIALIZATION_STATE_MISMATCH")
                except Exception as error:
                    context.persistable = False
                    context.reason = (str(error) if isinstance(error, IdentityUnavailable)
                                      else f"NPU_INITIALIZATION_UNAVAILABLE:{type(error).__name__}:{error}")
                    log.info("NPU inner compile cache bypass: %s", context.reason, exc_info=True)
            with request_scope(context):
                return inner(*args, **kwargs)

    bound._npu_request = context
    return bound
