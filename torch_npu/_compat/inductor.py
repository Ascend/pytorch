# Copyright (c) 2026, Huawei Technologies Co., Ltd

import functools
import sys

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
    "namespaced_compile_options",
    "compat_compile_option",
    "compat_global_compile_option",
    "compat_translate_compile_options",
    "compat_inject_legacy_compile_options",
    "triton_prologue_loads_all_kwargs",
    "install_triton_prologue_inputs_aliases",
    "triton_prologue_inputs_kwargs",
    "triton_prologue_inputs",
    "make_multi_template_buffer",
    "pop_compiled_triton_kernel_loader",
]


# upstream pytorch#200491/#200492 added the inline-kernel handover context;
# pytorch#200502 enabled module-level kernels by default on the 2.16 line.
@compat_impl(key="pop_compiled_triton_kernel_loader", ge=(2, 16))
def pop_compiled_triton_kernel_loader_upstream(kernel_name):
    from torch._inductor.runtime import triton_heuristics

    # Early 2.16 nightlies may predate the handover protocol.
    context = getattr(triton_heuristics, "compiled_kernels_for_load", None)
    compiled = context.get() if context is not None else None
    # Consume before the caller loads the kernel: loading its source can
    # re-enter the decorator while the wrapper's context is still active.
    return compiled.pop(kernel_name, None) if compiled is not None else None


@compat_impl(key="pop_compiled_triton_kernel_loader", lt=(2, 16))
def pop_compiled_triton_kernel_loader_local(kernel_name):
    """Older torch loads string kernels without a handover context."""
    return None


pop_compiled_triton_kernel_loader = compat_impl_container[
    "pop_compiled_triton_kernel_loader"
].resolve()


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
    from torch._inductor.codegen import cpp_utils

    upstream_device_to_aten = getattr(cpp_utils, "device_to_aten", None)
    if upstream_device_to_aten is not None:
        return upstream_device_to_aten(device_type)

    # Some 2.15 nightlies expose the override registry before this helper.
    if device_type == "meta":
        return "at::kMeta"

    from torch._inductor.codegen.common import get_device_op_overrides

    return get_device_op_overrides(device_type).aten_device_type()


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


# torch PR #194998 routes options named "<device>.<key>" to the device's
# registered device_custom_config. It is on torch main and in release/2.15,
# but 2.15 nightlies from before the merge are still in circulation (CI's
# 2.15 lane runs one), so CURRENT_VERSION cannot tell the spellings apart
# and the gate feature-probes instead. The probe imports
# torch._inductor.codegen.common, too heavy at import time for @compat_impl's
# import-time judgement, so the compat_* helpers below stay plain
# dual-path functions.
@functools.cache
def namespaced_compile_options() -> bool:
    import torch._inductor.codegen.common as codegen_common

    return hasattr(codegen_common, "get_compile_option_owner")


# COMPAT(pytorch#194998): "npu.<key>" once upstream routes namespaced
#   options; bare key on older torch with the injected entries below.
# CAN REMOVE when the upstream PR is required.
def compat_compile_option(key: str) -> str:
    return f"npu.{key}" if namespaced_compile_options() else key


# COMPAT(pytorch#194998): global defaults live in the npu options module
#   once namespaced, the injected core-config entry on older torch.
# CAN REMOVE the legacy branch when the upstream PR is required.
def compat_global_compile_option(key: str):
    if namespaced_compile_options():
        compile_options = sys.modules.get("torch_npu._inductor.compile_options")
        return getattr(compile_options, key, None)
    inductor_config = sys.modules.get("torch._inductor.config")
    return getattr(inductor_config, key, None)


# COMPAT(pytorch#194998): legacy bare spellings map onto namespaced keys;
#   identity on older torch.
# CAN REMOVE when legacy spellings are dropped.
_LEGACY_COMPILE_OPTIONS = {
    "npu_backend": "npu.npu_backend",
    "enable_shape_handling": "npu.enable_shape_handling",
    "shape_handling_configs": "npu.shape_handling_configs",
    "shape_handling_dict": "npu.shape_handling_dict",
}


def compat_translate_compile_options(options):
    if not options or not namespaced_compile_options():
        return options
    return {
        _LEGACY_COMPILE_OPTIONS.get(key, key): value
        for key, value in options.items()
    }


# COMPAT(pytorch#194998): get_config_copy patch that injects the option names
#   into torch._inductor.config so upstream apply_options accepts them.
# CAN REMOVE this function when the upstream PR is required.
def compat_inject_legacy_compile_options():
    if namespaced_compile_options():
        return

    from torch.utils._config_module import Config, ConfigModule, _ConfigEntry

    src_get_config_copy = ConfigModule.get_config_copy

    def new_get_config_copy(self) -> dict:
        ori_dict = src_get_config_copy(self)
        inductor_config = sys.modules.get("torch._inductor.config")
        if inductor_config is None or self is not inductor_config:
            return ori_dict
        if "npu_backend" not in ori_dict:
            ori_dict["npu_backend"] = "default"
            self._config["npu_backend"] = _ConfigEntry(
                Config(default="default", value_type=str),
                name="npu_backend",
            )

        if "enable_shape_handling" not in ori_dict:
            ori_dict["enable_shape_handling"] = False
            self._config["enable_shape_handling"] = _ConfigEntry(
                Config(default=False, value_type=bool),
                name="enable_shape_handling",
            )

        if "shape_handling_configs" not in ori_dict:
            ori_dict["shape_handling_configs"] = []
            self._config["shape_handling_configs"] = _ConfigEntry(
                Config(default=[], value_type=list),
                name="shape_handling_configs",
            )

        if "shape_handling_dict" not in ori_dict:
            ori_dict["shape_handling_dict"] = None
            self._config["shape_handling_dict"] = _ConfigEntry(
                Config(default=None, value_type=dict),
                name="shape_handling_dict",
            )
        return ori_dict

    ConfigModule.get_config_copy = new_get_config_copy


# upstream pytorch#00b578d (first in torch 2.16) reworked prologue fusion:
#   * TritonTemplate / TritonTemplateKernel take `prologue_loads_all_named_inputs`
#     instead of `prologue_loads_all_inputs`;
#   * a template kernel keeps `load_input_fusion_allowed_inputs` /
#     `load_input_fused_inputs` (+ store-output counterparts) instead of
#     `prologue_supported_inputs` / `prologue_fused_inputs`;
#   * Triton template buffers and callers take
#     `load_input_fusion_allowed_inputs` / `store_output_fusion_allowed_inputs`
#     instead of a single `allowed_prologue_inps`.
# CAN REMOVE this block when MIN_SUPPORTED >= (2, 16).
@compat_impl(key="triton_prologue_loads_all_kwargs", lt=(2, 16))
def triton_prologue_loads_all_kwargs_local(prologue_loads_all_inputs: bool) -> dict:
    return {"prologue_loads_all_inputs": prologue_loads_all_inputs}


@compat_impl(key="triton_prologue_loads_all_kwargs", ge=(2, 16))
def triton_prologue_loads_all_kwargs_upstream(prologue_loads_all_inputs: bool) -> dict:
    return {"prologue_loads_all_named_inputs": prologue_loads_all_inputs}


triton_prologue_loads_all_kwargs = compat_impl_container[
    "triton_prologue_loads_all_kwargs"
].resolve()


@compat_impl(key="install_triton_prologue_inputs_aliases", lt=(2, 16))
def install_triton_prologue_inputs_aliases_local(kernel) -> None:
    """torch < 2.16: the kernel already exposes the pre-rename attributes."""
    return


@compat_impl(key="install_triton_prologue_inputs_aliases", ge=(2, 16))
def install_triton_prologue_inputs_aliases_upstream(kernel) -> None:
    """torch >= 2.16: alias the pre-rename prologue attributes so the NPU
    template kernel code stays version agnostic.  The aliases share the
    upstream OrderedSets, which upstream only mutates in place, so they stay
    in sync (including `_finalize_fusion_allowed_inputs`'s ``-=`` trimming).
    """
    kernel.prologue_supported_inputs = kernel.load_input_fusion_allowed_inputs
    kernel.prologue_fused_inputs = kernel.load_input_fused_inputs


install_triton_prologue_inputs_aliases = compat_impl_container[
    "install_triton_prologue_inputs_aliases"
].resolve()


@compat_impl(key="triton_prologue_inputs_kwargs", lt=(2, 16))
def triton_prologue_inputs_kwargs_local(allowed_prologue_inps) -> dict:
    return {"allowed_prologue_inps": allowed_prologue_inps}


@compat_impl(key="triton_prologue_inputs_kwargs", ge=(2, 16))
def triton_prologue_inputs_kwargs_upstream(allowed_prologue_inps) -> dict:
    return {"load_input_fusion_allowed_inputs": allowed_prologue_inps}


triton_prologue_inputs_kwargs = compat_impl_container[
    "triton_prologue_inputs_kwargs"
].resolve()


# `None` means the object carries no prologue-fusion gating (e.g. an
# ExternKernel caller), mirroring the historical ``hasattr`` check.
@compat_impl(key="triton_prologue_inputs", lt=(2, 16))
def triton_prologue_inputs_local(obj):
    return getattr(obj, "allowed_prologue_inps", None)


@compat_impl(key="triton_prologue_inputs", ge=(2, 16))
def triton_prologue_inputs_upstream(obj):
    return getattr(obj, "load_input_fusion_allowed_inputs", None)


triton_prologue_inputs = compat_impl_container["triton_prologue_inputs"].resolve()


# MultiTemplateBuffer.__init__ takes the split load/store fusion sets on
#   torch >= 2.16 and a single allowed_prologue_inps set before that.  The
#   upstream change is feature-probed instead of version-gated because 2.15
#   nightlies in circulation predate the merge, and importing torch._inductor.ir
#   at compat import time is too heavy for a compat_impl judgement.
@functools.cache
def multi_template_buffer_splits_fusion_sets() -> bool:
    import inspect

    import torch._inductor.ir as ir

    return (
        "store_output_fusion_allowed_inputs"
        in inspect.signature(ir.MultiTemplateBuffer.__init__).parameters
    )


def make_multi_template_buffer(layout, input_nodes, choice_timings_fn, choices):
    """Build an ir.MultiTemplateBuffer with the prologue-fusion sets the
    installed torch expects.  We take the union of the allowed prologue inputs
    from all choices; benchmark fusion then skips choices that don't support
    the whole union.
    """
    import torch._inductor.ir as ir
    from torch._inductor.select_algorithm import TritonTemplateCaller
    from torch.utils._ordered_set import OrderedSet

    if multi_template_buffer_splits_fusion_sets():
        load_fusion_inputs: OrderedSet[str] = OrderedSet()
        store_fusion_inputs: OrderedSet[str] = OrderedSet()
        for choice in choices:
            if isinstance(choice, TritonTemplateCaller):
                load_fusion_inputs |= choice.load_input_fusion_allowed_inputs
                store_fusion_inputs |= choice.store_output_fusion_allowed_inputs
        return ir.MultiTemplateBuffer(
            layout,
            input_nodes,
            choice_timings_fn,
            choices,
            load_fusion_inputs,
            store_fusion_inputs,
        )

    prologue_inputs: OrderedSet[str] = OrderedSet()
    for choice in choices:
        if isinstance(choice, TritonTemplateCaller):
            prologue_inputs |= getattr(choice, "allowed_prologue_inps", OrderedSet())
    return ir.MultiTemplateBuffer(
        layout, input_nodes, choice_timings_fn, choices, prologue_inputs
    )
