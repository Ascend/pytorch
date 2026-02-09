import os

if os.getenv('TORCHINDUCTOR_NPU_BACKEND', 'default') == 'mlir':
    try:
        import torch_mlir
        from torch_mlir import ir
    except:
        raise ImportError("torch_mlir is not installed, install it first.")
    from .ascend_npu_ir.ascend_npu_ir.npu import npu_inductor_plugin
else:
    import torch
    from torch._dynamo.device_interface import register_interface_for_device, get_interface_for_device
    from torch._inductor import lowering as inductor_lowering
    from torch._inductor.choices import InductorChoices
    from torch._inductor.codegen.common import register_backend_for_device, register_device_op_overrides
    from torch._inductor.runtime import autotune_cache
    from torch_npu.npu import device_count
    from torch_npu.utils._dynamo_device import NpuInterface, current_device, set_device
    from torch_npu.utils._inductor import NPUDeviceOpOverrides

    from . import config as npu_config
    from . import codegen
    from .npu_fusion_attention_graph import register_fa_pass
    from .config import aggresive_autotune, num_vector_core, set_compile_threads, disable_comprehensive_padding
    from .config import log as npulog
    from .decomposition import _register_npu_inductor_decompositons
    from .lowering import make_reduction, npu_make_fallback
    from .npu_choices import should_use_persistent_reduction
    from .npu_device import NewNPUDeviceOpOverrides
    from .runtime import _load_cached_autotuning
    from .utils import get_current_raw_stream, patch_is_gpu, patch_has_triton, disable_foreach
    from .codecache import patch_aot_code_compiler_compile, patch_cache_base_get_system
    from .cpp_builder import patch_get_optimization_cflags

    set_compile_threads()
    disable_comprehensive_padding()


    def _inductor_register_backend_for_device():
        from .codegen.scheduling import NPUTritonScheduling
        from .codegen.wrapper import NPUWrapperCodeGen
        from .codegen.cpp_wrapper import CppWrapperNpu
        register_backend_for_device('npu', NPUTritonScheduling, NPUWrapperCodeGen, CppWrapperNpu)


    _inductor_register_backend_for_device()


    def _inductor_register_device_op_overrides():
        register_device_op_overrides('npu', NewNPUDeviceOpOverrides())


    _inductor_register_device_op_overrides()

    device = get_interface_for_device("npu")

    inductor_lowering.make_reduction = make_reduction
    inductor_lowering.make_fallback = npu_make_fallback


    def patch_torch_for_aoti():
        from .graph import patch_codegen_with_cpp_wrapper
        from .cpp_builder import patch_get_cpp_torch_device_options
        from .codegen.cpp_utils import patch_device_to_aten
        from .utils import patch_is_same_tensor
        from .fx_passes.joint_graph import patch_constant_fold_uniform_value
        from .ir import patch_fallback_kernel_codegen

        patch_codegen_with_cpp_wrapper()
        patch_get_cpp_torch_device_options()
        patch_device_to_aten()
        patch_is_same_tensor()
        patch_constant_fold_uniform_value()
        patch_fallback_kernel_codegen()

        patch_aot_code_compiler_compile()
        if os.environ.get("PRE_GRAPH_OPTIMIZER") == "1":
            from .fx_passes.graph_match_pass import pre_grad_custom_pass_fuc
            pre_grad_custom_pass_fuc()
        if os.environ.get("POST_GRAD_GRAPH_OPTIMIZER") == "1":
            from .fx_passes.graph_match_pass import post_grad_custom_pass_fuc
            post_grad_custom_pass_fuc()



    if os.environ.get("DISABLE_AOTI_PATCH", "0") != "1":
        patch_torch_for_aoti()


    if npu_config.dump_fx_graph:
        from .codegen.ir_fx import _patch_npu_inductor_ir

        _patch_npu_inductor_ir()

    if npu_config.dump_fx_graph:
        from .lowering_fx import _register_npu_inductor_fallbacks
    else:
        from .lowering import _register_npu_inductor_fallbacks

    _register_npu_inductor_fallbacks()
    _register_npu_inductor_decompositons()


    # register fx_pass should be put behind of _register_npu_inductor_decompositons
    def _replace_benchmark_all_configs():
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        from .npu_triton_heuristics import benchmark_all_configs
        CachingAutotuner.benchmark_all_configs = benchmark_all_configs


    if (aggresive_autotune):
        _replace_benchmark_all_configs()
        import os

        os.environ["TRITON_BENCH_METHOD"] = "npu"

    InductorChoices.should_use_persistent_reduction = should_use_persistent_reduction
    autotune_cache._load_cached_autotuning = _load_cached_autotuning


    def patch_device_override_func():
        def get_device_op_overrides_patch(device_name: str):
            def register_cpu_backend():
                from torch._inductor.codegen import cpu_device_op_overrides

                return

            def register_mps_backend():
                from torch._inductor.codegen import mps_device_op_overrides

                return

            backend_factory = {"cpu": register_cpu_backend, "mps": register_mps_backend}

            if device_name not in torch._inductor.codegen.common.device_op_overrides_dict:
                if device_name not in backend_factory:
                    raise ValueError("backend not found: ", device_name)
                backend_factory[device_name]()

            return torch._inductor.codegen.common.device_op_overrides_dict[device_name]

        torch._inductor.graph.get_device_op_overrides = get_device_op_overrides_patch

    register_fa_pass()
    patch_cache_base_get_system()
    patch_is_gpu()
    patch_has_triton()
    disable_foreach()
    patch_get_optimization_cflags()
    patch_device_override_func()
