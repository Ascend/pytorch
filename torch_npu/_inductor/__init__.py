
import os
ORG_AUTOLOAD = os.getenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
from torch._inductor.async_compile import AsyncCompile
AsyncCompile.warm_pool()
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = ORG_AUTOLOAD

# all backends need register npu/cpu/mps device_op_overrides
from .codegen.common import patch_get_device_op_overrides
patch_get_device_op_overrides()

if os.getenv('TORCHINDUCTOR_NPU_BACKEND', 'default') == 'mlir':
    try:
        import torch_mlir
        from torch_mlir import ir
    except:
        raise ImportError("torch_mlir is not installed, install it first.")
    from .ascend_npu_ir.ascend_npu_ir.npu import npu_inductor_plugin
    from .ascend_npu_ir.ascend_npu_ir.npu import torch_mlir_patch
    
elif os.getenv('TORCHINDUCTOR_NPU_BACKEND', 'default') == 'dvm':
    from .ascend_npu_ir.ascend_npu_ir.npu import npu_inductor_plugin
    from .dvm import mlir_fusion
else:
    import os

    import torch
    from torch._dynamo.device_interface import get_interface_for_device
    from torch._inductor import lowering as inductor_lowering
    from torch._inductor.codegen.common import register_backend_for_device, register_device_op_overrides

    from . import config as npu_config
    from . import codegen
    from .fx_passes.pattern_match.npu_fusion_attention_graph import register_fa_pass
    from .config import (
        aggresive_autotune, num_vector_core,
        disable_comprehensive_padding, max_precompiled_thread_num
    )
    from .config import log as npulog
    from .codegen._sizevars import patch_simplify
    from .codegen.ir import patch_num_split, patch_loop_body, patch_indexing
    from .codegen.triton import patch_gen_common_triton_ext_imports, patch_triton_scheduling, patch_is_compatible
    from .decomposition import _register_npu_inductor_decompositons
    from .graph import patch_count_bytes, patch_codegen_with_cpp_wrapper, patch_run_node
    from .ir import patch_fallback_kernel_codegen
    from .lowering import make_reduction
    from .runtime import (
        patch_load_cached_autotuning,
        patch_create_device_properties,
        patch_triton_heuristics_cached_autotune
    )
    from .choices import NPUInductorChoices
    from .utils import (
        patch_is_gpu,
        patch_has_triton,
        disable_foreach,
        patch_get_first_incompatible_cudagraph_node
    )
    from .codecache import patch_aot_code_compiler_compile, patch_cache_base_get_system
    from .cpp_builder import patch_get_cpp_torch_device_options
    from .codegen.cpp_utils import patch_device_to_aten
    from .scheduler import patch_scheduler
    from .shape_handling import NPUShapeHandling, patch_shape_handling
    from .async_compile import patch_async_compile
    from .autotune_process import patch_tuning_process, patch_tuning_process_pool
    from .select_algorithm import patch_algorithm_selector
    from .fx_passes import patch_pattern_mm_plus_mm
    from .fx_passes.graph_match_pass import pre_grad_custom_pass_fuc, post_grad_custom_pass_fuc
    from .fx_passes.joint_graph import patch_constant_fold_uniform_value
    from .kernel import (
        _register_npu_inductor_mm,
        _register_npu_inductor_addmm,
        _register_npu_inductor_bmm,
        _register_npu_inductor_grouped_mm,
        _register_npu_inductor_flex_attention,
        _validate_device,
    )
    from .cpp_builder import patch_get_optimization_cflags
    from torch.nn.attention import flex_attention
    flex_attention._validate_device = _validate_device

    disable_comprehensive_padding()


    def _inductor_register_backend_for_device():
        from .codegen.npu_combined_scheduling import NPUCombinedScheduling
        from .codegen.wrapper import NPUWrapperCodeGen
        from .codegen.cpp_wrapper import CppWrapperNpu
        register_backend_for_device('npu', NPUCombinedScheduling, NPUWrapperCodeGen, CppWrapperNpu)


    _inductor_register_backend_for_device()

    device = get_interface_for_device("npu")

    inductor_lowering.make_reduction = make_reduction

    patch_codegen_with_cpp_wrapper()
    patch_get_cpp_torch_device_options()
    patch_device_to_aten()
    patch_constant_fold_uniform_value()
    patch_fallback_kernel_codegen()
    patch_aot_code_compiler_compile()


    if npu_config.dump_fx_graph:
        from .codegen.ir_fx import _patch_npu_inductor_ir

        _patch_npu_inductor_ir()

    from .lowering import _register_npu_inductor_fallbacks

    _register_npu_inductor_fallbacks()
    _register_npu_inductor_decompositons()
    _register_npu_inductor_mm()
    _register_npu_inductor_addmm()
    _register_npu_inductor_bmm()
    _register_npu_inductor_grouped_mm()
    _register_npu_inductor_flex_attention()
    patch_pattern_mm_plus_mm()
    patch_algorithm_selector()
    patch_tuning_process()
    patch_tuning_process_pool()
    patch_async_compile()
    patch_scheduler()
    patch_gen_common_triton_ext_imports()
    patch_simplify()
    patch_num_split()
    patch_loop_body()
    patch_indexing()
    patch_triton_scheduling()
    patch_is_compatible()

    patch_create_device_properties()
    patch_load_cached_autotuning()
    patch_triton_heuristics_cached_autotune()

    pre_grad_custom_pass_fuc()
    post_grad_custom_pass_fuc()

    # register fx_pass should be put behind of _register_npu_inductor_decompositons
    def _replace_benchmark_all_configs():
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        from .runtime.triton_heuristics import benchmark_all_configs, _benchmark_all_configs
        CachingAutotuner._benchmark_all_configs = _benchmark_all_configs
        CachingAutotuner.benchmark_all_configs = benchmark_all_configs


    def _replace_precompile():
        from .runtime.triton_heuristics import precompile_parallel, NPUCachingAutotuner
        NPUCachingAutotuner.precompile = precompile_parallel


    if (aggresive_autotune):
        _replace_benchmark_all_configs()

    if (max_precompiled_thread_num > 1):
        _replace_precompile()

    torch._inductor.virtualized.V.set_choices_handler(NPUInductorChoices())

    register_fa_pass()
    patch_cache_base_get_system()
    patch_count_bytes()
    patch_run_node()
    patch_is_gpu()
    patch_has_triton()
    disable_foreach()
    patch_get_first_incompatible_cudagraph_node()
    patch_get_optimization_cflags()


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
