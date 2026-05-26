import os

# All backends need npu/cpu/mps device_op_overrides.
from .codegen.common import register_device_op_overrides_npu, patch_cache_base_get_system
from .graph import patch_codegen_with_cpp_wrapper
from .utils import patch_has_triton, patch_device_supports_tma, patch_is_gpu, get_current_raw_stream
from torch._inductor.lowering import make_fallback as _ori_make_fallback
register_device_op_overrides_npu()
patch_has_triton()
patch_is_gpu()
patch_device_supports_tma()
patch_codegen_with_cpp_wrapper()
patch_cache_base_get_system()


def _get_backend() -> str:
    return os.getenv("TORCHINDUCTOR_NPU_BACKEND", "default")

def _load_mlir_backend():
    import torch
    try:
        import torch_mlir
        from torch_mlir import ir
    except ImportError as err:
        raise ImportError("torch_mlir is not installed, install it first.") from err
    global _ori_make_fallback
    torch._inductor.lowering.make_fallback = _ori_make_fallback
    from .ascend_npu_ir.ascend_npu_ir.npu import npu_inductor_plugin
def _load_triton_backend():
    import os
    import torch
    has_triton = torch.utils._triton.has_triton()
    if not has_triton:
        return
    from torch._dynamo.device_interface import (
        get_interface_for_device,
        register_interface_for_device,
    )
    from torch._inductor import lowering as inductor_lowering
    from torch._inductor.choices import InductorChoices
    from torch._inductor.codegen.common import (
        register_backend_for_device,
        register_device_op_overrides,
    )
    from torch._inductor.runtime import autotune_cache
    from torch_npu.npu import device_count
    from torch_npu.utils._dynamo_device import current_device, NpuInterface, set_device
    from torch_npu.utils._inductor import NPUDeviceOpOverrides

    from . import codegen, config as npu_config
    from .codecache import patch_aot_code_compiler_compile
    from .config import aggresive_autotune, log as npulog, num_vector_core
    from .decomposition import _register_npu_inductor_decompositons
    from .lowering import make_reduction, npu_make_fallback
    from .npu_choices import should_use_persistent_reduction
    from .npu_device import NewNPUDeviceOpOverrides
    from .npu_fusion_attention_graph import register_fa_pass
    from .runtime import _load_cached_autotuning
    from .utils import (
        disable_foreach,
        patch_fx_node_is_input_dependent_cudagraph_unsafe,
    )

    def _inductor_register_backend_for_device():
        from .codegen.cpp_wrapper import CppWrapperNpu
        from .codegen.scheduling import NPUTritonScheduling
        from .codegen.wrapper import NPUWrapperCodeGen

        register_backend_for_device(
            "npu", NPUTritonScheduling, NPUWrapperCodeGen, CppWrapperNpu
        )

    _inductor_register_backend_for_device()

    device = get_interface_for_device("npu")

    inductor_lowering.make_reduction = make_reduction
    inductor_lowering.make_fallback = npu_make_fallback

    def patch_torch_for_aoti():
        from .codegen.cpp_utils import patch_device_to_aten
        from .cpp_builder import patch_get_cpp_torch_device_options
        from .fx_passes.joint_graph import patch_constant_fold_uniform_value
        from .graph import patch_codegen_with_cpp_wrapper
        from .ir import patch_fallback_kernel_codegen
        from .utils import patch_is_same_tensor

        patch_codegen_with_cpp_wrapper()
        patch_get_cpp_torch_device_options()
        patch_device_to_aten()
        patch_is_same_tensor()
        patch_constant_fold_uniform_value()
        patch_fallback_kernel_codegen()

        from .ir import patch_extern_kernel_codegen_size_asserts

        patch_extern_kernel_codegen_size_asserts()

        patch_aot_code_compiler_compile()

    if os.environ.get("DISABLE_AOTI_PATCH", "0") != "1":
        patch_torch_for_aoti()

    if npu_config.dump_fx_graph:
        from .codegen.ir_fx import _patch_npu_inductor_ir

        _patch_npu_inductor_ir()

    from .lowering import (
        _enable_full_lowering_fallback,
        _register_npu_inductor_fallbacks,
    )

    _register_npu_inductor_decompositons()

    if npu_config.enable_full_lowering_fallback.strip() == "allfallback":
        _enable_full_lowering_fallback()
    else:
        _register_npu_inductor_fallbacks()

    # register fx_pass should be put behind of _register_npu_inductor_decompositons
    def _replace_benchmark_all_configs():
        from torch_npu._compat.inductor import get_CachingAutotuner

        CachingAutotuner = get_CachingAutotuner()
        from .npu_triton_heuristics import benchmark_all_configs

        CachingAutotuner.benchmark_all_configs = benchmark_all_configs

    if aggresive_autotune:
        _replace_benchmark_all_configs()
        import os

        os.environ["TRITON_BENCH_METHOD"] = "npu"

    InductorChoices.should_use_persistent_reduction = should_use_persistent_reduction
    autotune_cache._load_cached_autotuning = _load_cached_autotuning

    register_fa_pass()
    disable_foreach()
    patch_fx_node_is_input_dependent_cudagraph_unsafe()
    os.environ["TORCHINDUCTOR_COMPREHENSIVE_PADDING"] = "0"
    torch._inductor.config.comprehensive_padding = False
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    torch._inductor.config.compile_threads = 1

_BACKEND_LOADERS = {
    "mlir": _load_mlir_backend,
    "default": _load_triton_backend,
}


def _load_backend():
    backend = _get_backend()
    loader = _BACKEND_LOADERS.get(backend, _load_triton_backend)
    loader()
    from ..utils._dynamo import _InductorNpuRegistry
    _InductorNpuRegistry._loaded_backend = backend

_load_backend()
