# Copyright (c) 2026, Huawei Technologies Co., Ltd

from torch_npu._compat.version import CURRENT_VERSION


# COMPAT(>= 2.15): upstream pytorch#190615 added DeviceOpOverrides and
#   DeviceInterface hooks for C++ wrapper routing and device properties.
# CAN REMOVE when MIN_SUPPORTED >= (2, 15): remove the legacy patch helpers.
def patch_codegen_with_cpp_wrapper():
    """
    patch codegen for cpp wrapper, add npu for codegen_with_cpp_wrapper function

    """
    if CURRENT_VERSION >= (2, 15):
        return

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


# COMPAT(>= 2.15): upstream pytorch#190615 added
#   DeviceInterface.get_multi_processor_count.
# CAN REMOVE when MIN_SUPPORTED >= (2, 15): stop patching DeviceProperties.create.
def patch_create_device_properties():
    if CURRENT_VERSION >= (2, 15):
        return

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


# COMPAT(>= 2.15): upstream pytorch#193908 added cache system metadata hooks.
# CAN REMOVE when MIN_SUPPORTED >= (2, 15): stop replacing CacheBase.get_system.
def patch_cache_base_get_system():
    if CURRENT_VERSION >= (2, 15):
        return

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


# COMPAT(>= 2.15): upstream pytorch#193904 removed DEVICE_TO_ATEN and
#   routes ATen device mapping through DeviceOpOverrides.
# CAN REMOVE when MIN_SUPPORTED >= (2, 15): stop registering the legacy mapping.
def patch_device_to_aten():
    if CURRENT_VERSION >= (2, 15):
        return

    from torch._inductor import codegen

    codegen.cpp_utils.DEVICE_TO_ATEN["npu"] = "at::kPrivateUse1"


# COMPAT(>= 2.15): upstream pytorch#193904 moved ATen device mapping to
#   DeviceOpOverrides and added device_to_aten.
# CAN REMOVE when MIN_SUPPORTED >= (2, 15): use upstream device_to_aten directly.
def device_to_aten(device_type: str) -> str:
    if CURRENT_VERSION >= (2, 15):
        from torch._inductor.codegen.cpp_utils import device_to_aten as upstream_device_to_aten

        return upstream_device_to_aten(device_type)
    from torch._inductor.codegen.cpp_utils import DEVICE_TO_ATEN

    return DEVICE_TO_ATEN[device_type]
