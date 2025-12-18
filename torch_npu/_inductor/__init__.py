import os

import torch
from torch._dynamo.device_interface import register_interface_for_device, get_interface_for_device
from torch._inductor import cpp_builder
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


def patch_get_optimization_cflags(
    cpp_compiler: str, min_optimize: bool = False
) -> list[str]: 
    if _IS_WINDOWS: 
        return ["O1" if min_optimize else "O2"]
    else: 
        wrapper_opt_level = config.aot_inductor.compile_wrapper_opt_level
        cflags = (
            ["O0", "g"]
            if config.aot_inductor.debug_compile
            else [wrapper_opt_level if min_optimize else "O3", "DNDEBUG"] 
        )
        cflags += _get_ffast_math_flags()
        cflags.append("fno-finite-math-only")
        if not config.cpp.enable_unsafe_math_opt_flag:
            cflags.append("fno-unsafe-math-optimizations")
        cflags.append(f"ffp-contract={config.cpp.enable_floating_point_contract_flag}") 
        
        if sys.platform != "darwin":
            # on macos, unknown argument: '-fno-tree-loop-vectorize' 
            if _is_gcc(cpp_compiler):
                cflags.append("fno-tree-loop-vectorize")
            # -march=native is unrecognized option on M1 
            if not config.is_fbcode(): 
                if platform.machine() == "ppc64le":
                    cflags.append("mcpu=native")

        return cflags

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
    from torch._inductor.triton_heuristics import CachingAutotuner
    from .npu_triton_heuristics import benchmark_all_configs
    CachingAutotuner.benchmark_all_configs = benchmark_all_configs


if (aggresive_autotune):
    _replace_benchmark_all_configs()
    import os

    os.environ["TRITON_BENCH_METHOD"] = "npu"

InductorChoices.should_use_persistent_reduction = should_use_persistent_reduction
autotune_cache._load_cached_autotuning = _load_cached_autotuning
cpp_builder._get_optimization_cflags = patch_get_optimization_cflags

register_fa_pass()
patch_cache_base_get_system()
patch_is_gpu()
patch_has_triton()
disable_foreach()

