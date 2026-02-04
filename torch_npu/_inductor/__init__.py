
import os

if os.getenv('TORCHINDUCTOR_NPU_BACKEND', 'default') == 'mlir':
    try:
        import torch_mlir
        from torch_mlir import ir
    except:
        raise ImportError("torch_mlir is not installed, install it first.")
    from .ascend_npu_ir.build_ext import build_ascend_npu_ir_ext, set_torch_npu_library_path
    _has_inited = False
    if not _has_inited:
        _has_inited = True
        build_ascend_npu_ir_ext()
    set_torch_npu_library_path()
    from .ascend_npu_ir.ascend_npu_ir.npu import npu_inductor_plugin
else:
    import os

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
    from .config import (
        aggresive_autotune, num_vector_core, set_compile_threads, 
        disable_comprehensive_padding, max_precompiled_thread_num
    )
    from .config import log as npulog
    from .decomposition import _register_npu_inductor_decompositons
    from .graph import patch_count_bytes
    from .lowering import make_reduction, npu_make_fallback
    from .npu_choices import should_use_persistent_reduction
    from .npu_device import NewNPUDeviceOpOverrides
    from .runtime import _load_cached_autotuning
    from .utils import get_current_raw_stream, patch_is_gpu, patch_has_triton, disable_foreach
    from .codecache import patch_aot_code_compiler_compile, patch_cache_base_get_system
    from .scheduler import patch_scheduler
    from .shape_handling import NPUShapeHandling, patch_shape_handling
    from .async_compile import patch_async_compile
    from .autotune_process import patch_tuning_process, patch_tuning_process_pool
    from .select_algorithm import patch_algorithm_selector
    from .fx_passes import patch_pattern_mm_plus_mm
    from .kernel import (
        _register_npu_inductor_mm,
        _register_npu_inductor_addmm,
        _register_npu_inductor_bmm,
        _register_npu_inductor_grouped_mm,
    )

    set_compile_threads()
    disable_comprehensive_padding()


    def _inductor_register_backend_for_device():
        from .codegen.npu_combined_scheduling import NPUCombinedScheduling
        from .codegen.wrapper import NPUWrapperCodeGen
        from .codegen.cpp_wrapper import CppWrapperNpu
        register_backend_for_device('npu', NPUCombinedScheduling, NPUWrapperCodeGen, CppWrapperNpu)


    _inductor_register_backend_for_device()


    def _inductor_register_device_op_overrides():
        from torch._inductor.codegen import cpu_device_op_overrides
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

        from .fx_passes.graph_match_pass import pre_grad_custom_pass_fuc 
        pre_grad_custom_pass_fuc() 
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
    _register_npu_inductor_mm()
    _register_npu_inductor_addmm()
    _register_npu_inductor_bmm()
    _register_npu_inductor_grouped_mm()
    patch_pattern_mm_plus_mm()
    patch_algorithm_selector()
    patch_tuning_process()
    patch_tuning_process_pool()
    patch_async_compile()
    patch_scheduler()


    # register fx_pass should be put behind of _register_npu_inductor_decompositons
    def _replace_benchmark_all_configs():
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        from .npu_triton_heuristics import benchmark_all_configs, _benchmark_all_configs
        CachingAutotuner._benchmark_all_configs = _benchmark_all_configs
        CachingAutotuner.benchmark_all_configs = benchmark_all_configs


    def _replace_precompile():
        from .npu_triton_heuristics import precompile_parallel, NPUCachingAutotuner
        NPUCachingAutotuner.precompile = precompile_parallel


    if (aggresive_autotune):
        _replace_benchmark_all_configs()

    if (max_precompiled_thread_num > 1):
        _replace_precompile()

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
    patch_count_bytes()
    patch_is_gpu()
    patch_has_triton()
    disable_foreach()
    patch_device_override_func()


    def add_additional_op():
        from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
        from torch._inductor.ops_handler import OpsHandler
        from torch._inductor.utils import register_op_dtype_propagation_rules

        def cat_insert_slice(self, dst, src, offset, size, output_size):
            return self._default("cat_insert_slice", (dst, src, offset, size, output_size), {})

        OpsHandler.cat_insert_slice = cat_insert_slice
        register_op_dtype_propagation_rules("cat_insert_slice", ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, None)

        def cat_store(self, dst, src, size, store_offset_index, output_buffer_index):
            return self._default("cat_store", (dst, src, size, store_offset_index, output_buffer_index), {})

        OpsHandler.cat_store = cat_store
        register_op_dtype_propagation_rules("cat_store", ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, None)

        def index_select(self, name, index, indirect_var, set_indirect, bound, index_select_type):
            return self._default("index_select", (name, index, indirect_var, set_indirect, bound, index_select_type), {})

        OpsHandler.index_select = index_select
        register_op_dtype_propagation_rules("index_select", ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, None)

        def gather_template(self, name, index, indirect_var, set_indirect, index_boundary):
            return self._default("gather_template", (name, index, indirect_var, set_indirect, index_boundary), {})

        OpsHandler.gather_template = gather_template
        register_op_dtype_propagation_rules("gather_template", ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, None)

        def indexput_template(self, name, index, value, indirect_var, boundary):
            return self._default("indexput_template", (name, index, value, indirect_var, boundary), {})

        OpsHandler.indexput_template = indexput_template
        register_op_dtype_propagation_rules("indexput_template", ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, None)

        def scatter_template(self, name, index, value, indirect_var, boundary):
            return self._default("scatter_template", (name, index, value, indirect_var, boundary), {})

        OpsHandler.scatter_template = scatter_template
        register_op_dtype_propagation_rules("scatter_template", ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, None)


    add_additional_op()

