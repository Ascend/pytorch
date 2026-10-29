# Copyright (c) 2026, Huawei Technologies Co., Ltd

import functools

from torch_npu._compat._impl import compat_impl, compat_impl_container

__all__ = [
    "patch_codegen_with_cpp_wrapper",
    "patch_create_device_properties",
    "patch_cache_base_get_system",
    "patch_device_to_aten",
    "device_to_aten",
    "has_triton",
    "patch_has_triton",
    "patch_inductor_has_triton_bindings",
    "invalidate_has_triton_cache",
    "get_cpp_torch_device_options",
    "patch_npu_cpp_device_options",
    "compat_attach_inductor_backend_init",
    "compat_prepare_inductor_backend_for_call",
    "compat_prepare_dvm_backend_on_wrapper_init",
]


# upstream pytorch#190615 added DeviceOpOverrides and DeviceInterface hooks
# for C++ wrapper routing and device properties.
@compat_impl(key="patch_codegen_with_cpp_wrapper", ge=(2, 15))
def patch_codegen_with_cpp_wrapper_upstream():
    return


@compat_impl(key="patch_codegen_with_cpp_wrapper", lt=(2, 15))
def patch_codegen_with_cpp_wrapper_local():
    """
    patch codegen for cpp wrapper, add npu for codegen_with_cpp_wrapper function

    """
    import itertools
    import torch
    from torch._dynamo.utils import defake
    from torch._inductor import config, graph as inductor_graph, metrics
    from torch._inductor.utils import clone_preserve_strides
    from torch._inductor.virtualized import NullHandler, V
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.fx.node import Node

    GraphLowering = inductor_graph.GraphLowering

    def npu_codegen_with_cpp_wrapper(self) -> tuple[str, list[tuple[int, Node]]]:
        if any(device in self.device_types for device in ["cuda", "xpu", "npu"]):
            if config.triton.autotune_at_compile_time:
                # If autotune_at_compile_time is True, we can do the codegen in one-pass
                return self.codegen()
            else:
                # first pass
                self.cpp_wrapper = False
                compiled = self.compile_to_module().call

                def materialize(
                    x: torch.SymInt | torch.SymFloat | torch.Tensor,
                ) -> int | float | torch.Tensor:
                    if x is None:
                        return None
                    elif isinstance(x, (torch.SymInt, torch.SymFloat)):
                        # Need concrete value to run dynamic shapes and tune the result
                        return x.node.hint
                    elif isinstance(x, FakeTensor):
                        return defake(x)
                    else:
                        if not isinstance(x, torch.Tensor):
                            raise AssertionError(
                                "Unknown type when creating real inputs" + str(type(x))
                            )
                        return x

                tracing_context = torch._guards.TracingContext.try_get()
                if tracing_context is not None and not isinstance(
                    V.real_inputs, NullHandler
                ):
                    if tracing_context.output_strides:
                        tracing_context.output_strides.clear()

                    params_flat = [
                        param
                        for param in tracing_context.params_flat  # type: ignore[union-attr]
                        if param is not None
                    ]
                    real_inputs = [
                        materialize(x)
                        for x in itertools.chain(params_flat, V.real_inputs)
                    ]
                else:
                    # In the backward pass, V.real_inputs is not OrderedSet.
                    # Generating random inputs based on self.example_inputs sometimes can be problematic,
                    # e.g. illegal memory access. A comprehensive fix is to autotune in a separate process.
                    real_inputs = [
                        materialize(x)  # type:ignore[arg-type]
                        for x in (
                            self.example_inputs  # type:ignore[union-attr]
                            if isinstance(V.real_inputs, NullHandler)
                            else V.real_inputs
                        )
                    ]

                if self.mutated_inputs:
                    mutated_input_idxs = [
                        idx
                        for idx, name in enumerate(self.graph_inputs)
                        if name in self.mutated_inputs
                        and isinstance(real_inputs[idx], torch.Tensor)
                    ]
                    for idx in mutated_input_idxs:
                        # clone mutated Tensor inputs to avoid mutating them in
                        # the first pass of the CPP wrapper-based compilation, as
                        # this will lead to a side effect on the example inputs:
                        # e.g. if torch.compile(f)(x) if called on input-mutating
                        # f, the inputs x will be mutated twice in the process:
                        # once here, and again when running the compiled model;
                        # this will also lead to a numerically incorrect output
                        mutated_inp = real_inputs[idx]
                        if not isinstance(mutated_inp, torch.Tensor):
                            raise AssertionError
                        real_inputs[idx] = clone_preserve_strides(mutated_inp)
                        del mutated_inp

                with torch.utils._python_dispatch._disable_current_modes():
                    compiled(real_inputs)
                del real_inputs

                # second pass
                self.cpp_wrapper = True
                self.removed_buffers.clear()
                self.removed_operations.clear()
                self.inplaced_to_remove.clear()
                V.graph.sizevars.precomputed_replacements.clear()
                V.graph.sizevars.inv_precomputed_replacements.clear()
                metrics.reset()
                with config.patch({"triton.autotune_at_compile_time": False}):
                    return self.codegen()
        else:
            # cpu
            return self.codegen()

    GraphLowering.codegen_with_cpp_wrapper = npu_codegen_with_cpp_wrapper


patch_codegen_with_cpp_wrapper = compat_impl_container[
    "patch_codegen_with_cpp_wrapper"
].resolve()


# upstream pytorch#190615 added DeviceInterface.get_multi_processor_count.
@compat_impl(key="patch_create_device_properties", ge=(2, 15))
def patch_create_device_properties_upstream():
    return


@compat_impl(key="patch_create_device_properties", lt=(2, 15))
def patch_create_device_properties_local():

    import functools
    from torch._inductor.runtime.hints import DeviceProperties

    class NPUDeviceProperties(DeviceProperties):
        @classmethod
        @functools.lru_cache(None)
        def create(cls, device) -> DeviceProperties:
            import torch
            from torch._dynamo.device_interface import get_interface_for_device

            device_type = device.type
            if torch.version.hip and device_type == "cuda":
                device_type = "hip"
            device_interface = get_interface_for_device(device)
            props = device_interface.get_device_properties(device)
            try:
                multi_processor_count = props.vector_core_num
            except AttributeError:
                if device_type == "xpu":
                    multi_processor_count = props.gpu_subslice_count
                else:
                    raise
            return DeviceProperties(
                type=device_type,
                index=device.index,
                multi_processor_count=multi_processor_count,
                cc=device_interface.get_compute_capability(device),
                major=getattr(props, "major", None),
                regs_per_multiprocessor=getattr(props, "regs_per_multiprocessor", None),
                max_threads_per_multi_processor=getattr(
                    props, "max_threads_per_multi_processor", None
                ),
                warp_size=getattr(props, "warp_size", 32 if device_type != "cpu" else None),
            )

    DeviceProperties.create = NPUDeviceProperties.create


patch_create_device_properties = compat_impl_container[
    "patch_create_device_properties"
].resolve()


# upstream pytorch#193908 added cache system metadata hooks.
@compat_impl(key="patch_cache_base_get_system", ge=(2, 15))
def patch_cache_base_get_system_upstream():
    return


@compat_impl(key="patch_cache_base_get_system", lt=(2, 15))
def patch_cache_base_get_system_local():

    import hashlib
    import json
    from typing import Any, Dict

    import torch
    import torch_npu
    from torch._inductor.codecache import CacheBase

    @staticmethod
    def get_system():
        try:
            try:
                from triton.runtime.cache import triton_key
            except ImportError:
                from triton.compiler.compiler import triton_key
            # Use triton_key instead of triton.__version__ as the version
            # is not updated with each code change
            triton_version = triton_key()
        except ModuleNotFoundError:
            triton_version = None

        try:
            system: Dict[str, Any] = {
                "device": {"name": None},
                "version": {"triton": triton_version},
            }
            device_properties = torch_npu.npu.get_device_properties(
                torch_npu.npu.current_device()
            )
            if torch.version.cann is not None:
                system["device"]["name"] = device_properties.name
                system["version"]["cann"] = torch.version.cann
            elif torch.version.cuda is not None:
                system["device"]["name"] = device_properties.name
                system["version"]["cuda"] = torch.version.cuda
            else:
                system["device"]["name"] = device_properties.gcnArchName
                system["version"]["hip"] = torch.version.hip
        except (AssertionError, RuntimeError):
            # If device is not installed, none of the above config is relevant.
            system = {}

        system["hash"] = hashlib.sha256(
            json.dumps(system, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return system

    CacheBase.get_system = get_system


patch_cache_base_get_system = compat_impl_container[
    "patch_cache_base_get_system"
].resolve()


# upstream pytorch#193904 removed DEVICE_TO_ATEN and routes ATen device
# mapping through DeviceOpOverrides.
@compat_impl(key="patch_device_to_aten", ge=(2, 15))
def patch_device_to_aten_upstream():
    return


@compat_impl(key="patch_device_to_aten", lt=(2, 15))
def patch_device_to_aten_local():
    from torch._inductor import codegen

    codegen.cpp_utils.DEVICE_TO_ATEN["npu"] = "at::kPrivateUse1"


patch_device_to_aten = compat_impl_container["patch_device_to_aten"].resolve()


# upstream pytorch#193904 moved ATen device mapping to DeviceOpOverrides and
# added device_to_aten.
@compat_impl(key="device_to_aten", ge=(2, 15))
def device_to_aten_upstream(device_type: str) -> str:
    from torch._inductor.codegen.cpp_utils import device_to_aten as upstream_device_to_aten

    return upstream_device_to_aten(device_type)


@compat_impl(key="device_to_aten", lt=(2, 15))
def device_to_aten_local(device_type: str) -> str:
    from torch._inductor.codegen.cpp_utils import DEVICE_TO_ATEN

    return DEVICE_TO_ATEN[device_type]


device_to_aten = compat_impl_container["device_to_aten"].resolve()


# upstream pytorch#190324 (first in torch 2.14) resolves has_triton() through
#   the DeviceInterface registry and NpuInterface.is_triton_capable(), so the
#   NPU-local has_triton() and the patches installing it are pre-2.14 only.
#   has_triton is the _local sides' shared body, not a compat point: nothing
#   binds it on torch >= 2.14.
# CAN REMOVE this block when MIN_SUPPORTED >= (2, 14), together with the
#   re-exports in torch_npu.utils._dynamo, their call sites, and the unguarded
#   overrides in torch_npu.contrib.transfer_to_npu.
@functools.lru_cache(None)
def has_triton(*, include_cpu: bool = False) -> bool:
    from torch.utils._triton import has_triton_package

    if not has_triton_package():
        return False

    from torch._dynamo.device_interface import get_interface_for_device

    from torch_npu.utils._dynamo import _dynamo_register_interface_for_device

    def cpu_extra_check(device_interface):
        import triton.backends

        return "cpu" in triton.backends.backends

    triton_supported_devices = {
        "cuda": lambda di: True,
        "xpu": lambda di: True,
        "cpu": cpu_extra_check,
        "npu": lambda di: True,
    }

    _dynamo_register_interface_for_device()
    for device, extra_check in triton_supported_devices.items():
        if device == "cpu" and not include_cpu:
            continue
        device_interface = get_interface_for_device(device)
        if device_interface.is_available() and extra_check(device_interface):
            return True
    return False


@compat_impl(key="patch_has_triton", ge=(2, 14))
def patch_has_triton_upstream():
    """torch >= 2.14: upstream's registry-based has_triton() stays in place."""
    return


@compat_impl(key="patch_has_triton", lt=(2, 14))
def patch_has_triton_local():
    from torch.utils import _triton

    _triton.has_triton = has_triton


patch_has_triton = compat_impl_container["patch_has_triton"].resolve()


# Same upstream change as above: compile_fx and scheduler bind has_triton by
#   value at import time, so their copies need this separate patch, after
#   Inductor is imported.
@compat_impl(key="patch_inductor_has_triton_bindings", ge=(2, 14))
def patch_inductor_has_triton_bindings_upstream():
    """torch >= 2.14: the name is never rebound; every copy shares one cache."""
    return


@compat_impl(key="patch_inductor_has_triton_bindings", lt=(2, 14))
def patch_inductor_has_triton_bindings_local():
    from torch._inductor import compile_fx, scheduler

    scheduler.has_triton = has_triton
    compile_fx.has_triton = has_triton


patch_inductor_has_triton_bindings = compat_impl_container[
    "patch_inductor_has_triton_bindings"
].resolve()


# upstream's has_triton() is functools-cached and registry-based: a call
#   preceding NpuInterface registration caches a permanent False. Drop that
#   snapshot after registration; only helps callers that ask again (an
#   import-time gate stays degraded for the process).
@compat_impl(key="invalidate_has_triton_cache", ge=(2, 14))
def invalidate_has_triton_cache_upstream():
    from torch.utils import _triton

    cache_clear = getattr(_triton.has_triton, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


@compat_impl(key="invalidate_has_triton_cache", lt=(2, 14))
def invalidate_has_triton_cache_local():
    """torch < 2.14: torch_npu's own has_triton() registers the interface
    itself, so there is nothing to invalidate."""
    return


invalidate_has_triton_cache = compat_impl_container[
    "invalidate_has_triton_cache"
].resolve()


# COMPAT(>= 2.15): upstream pytorch#190614 added the
#   DeviceInterface.get_cpp_device_options hook.
# CAN REMOVE when MIN_SUPPORTED >= (2, 15): remove this fallback and its call
def get_cpp_torch_device_options(
    device_type: str, aot_mode: bool = False, compile_only: bool = False
):
    from torch_npu._inductor.cpp_builder import include_paths, library_paths

    npu = device_type == "npu"
    definitions = []
    include_dirs = include_paths(npu)
    cflags = []
    ldflags = []
    libraries_dirs = library_paths(npu)
    libraries = []
    passthough_args = []
    if npu:
        definitions.append("USE_NPU")
        libraries += ["torch_npu", "runtime", "ascendcl"]
        passthough_args += ["-DBUILD_LIBTORCH=ON -Wno-unused-function"]
    return (
        definitions,
        include_dirs,
        cflags,
        ldflags,
        libraries_dirs,
        libraries,
        passthough_args,
    )


@compat_impl(key="patch_npu_cpp_device_options", ge=(2, 15))
def patch_npu_cpp_device_options_upstream():
    return


@compat_impl(key="patch_npu_cpp_device_options", lt=(2, 15))
def patch_npu_cpp_device_options_local():
    from torch._inductor import cpp_builder

    cpp_builder.get_cpp_torch_device_options = get_cpp_torch_device_options


patch_npu_cpp_device_options = compat_impl_container[
    "patch_npu_cpp_device_options"
].resolve()


# torch PR #192896 (first in torch 2.15) fires the device module's
# _inductor_backend_init hook at every inductor compile entry while the
# device is unregistered. Both the hook attach (torch >= 2.15) and the
# historical wrapper-driven initialization (torch < 2.15) live here,
# version-isolated.


def _npu_inductor_backend_init():
    """Load and register the NPU backend through the torch device hook.

    torch >= 2.15 calls this at every inductor compile entry while the device
    is unregistered, so it must be idempotent.
    """
    from torch_npu.utils._dynamo import register_inductor_npu

    register_inductor_npu()


@compat_impl(key="compat_attach_inductor_backend_init", ge=(2, 15))
def compat_attach_inductor_backend_init_upstream(device_module):
    """Attach the device-module hook on torch >= 2.15."""
    device_module._inductor_backend_init = _npu_inductor_backend_init


@compat_impl(key="compat_attach_inductor_backend_init", lt=(2, 15))
def compat_attach_inductor_backend_init_local(device_module):
    """torch < 2.15 has no _inductor_backend_init hook: do nothing."""
    return


compat_attach_inductor_backend_init = compat_impl_container[
    "compat_attach_inductor_backend_init"
].resolve()


@compat_impl(key="compat_prepare_inductor_backend_for_call", ge=(2, 15))
def compat_prepare_inductor_backend_for_call_upstream(backend):
    """On torch >= 2.15 the upstream hook owns the first load; the wrapper
    switches an already initialized backend and installs shared integration.
    """
    from torch_npu.utils._dynamo import (
        _InductorNpuRegistry,
        _lazy_inductor_setup,
        register_inductor_npu,
    )

    loaded_backend = _InductorNpuRegistry._loaded_backend
    if loaded_backend is not None and loaded_backend != backend:
        register_inductor_npu()
    _lazy_inductor_setup()


@compat_impl(key="compat_prepare_inductor_backend_for_call", lt=(2, 15))
def compat_prepare_inductor_backend_for_call_local(backend):
    """On older torch the wrapper remains responsible for first-time
    initialization.
    """
    from torch_npu.utils._dynamo import _lazy_inductor_setup, register_inductor_npu

    register_inductor_npu()
    _lazy_inductor_setup()


compat_prepare_inductor_backend_for_call = compat_impl_container[
    "compat_prepare_inductor_backend_for_call"
].resolve()


@compat_impl(key="compat_prepare_dvm_backend_on_wrapper_init", ge=(2, 15))
def compat_prepare_dvm_backend_on_wrapper_init_upstream(backend):
    """torch >= 2.15 initializes lazily through the device hook."""
    return


@compat_impl(key="compat_prepare_dvm_backend_on_wrapper_init", lt=(2, 15))
def compat_prepare_dvm_backend_on_wrapper_init_local(backend):
    """Preserve the historical eager DVM initialization performed during
    wrapper construction on old torch.
    """
    from torch_npu.utils._dynamo import (
        _NpuBackendEnvScope,
        _lazy_inductor_setup,
        register_inductor_npu,
    )

    if backend != "dvm":
        return

    with _NpuBackendEnvScope(backend):
        register_inductor_npu()
        _lazy_inductor_setup()


compat_prepare_dvm_backend_on_wrapper_init = compat_impl_container[
    "compat_prepare_dvm_backend_on_wrapper_init"
].resolve()
