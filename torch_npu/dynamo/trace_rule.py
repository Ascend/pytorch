import torch
from torch._dynamo.variables import TorchInGraphFunctionVariable
from torch._dynamo.trace_rules import manual_torch_name_rule_map, SkipFunctionVariable
import torch._dynamo.variables.torch as torch_module
from torch._dynamo.utils import common_constant_types
import torch_npu

__all__ = []

torch_non_c_binding_in_graph_functions_npu = dict.fromkeys(
    [
        "torch.npu.current_stream",
        "torch.npu.default_stream",
        "torch.npu.stream",
        "torch.npu.set_stream",
        "torch_npu.npu.utils.synchronize",
        "torch.npu.current_device",
        "torch.npu.get_device_capability",
        "torch.npu.get_device_properties",
        "torch.npu.graphs.graph_pool_handle",
        "torch.npu.ipc_collect",
        "torch.npu.is_available",
        "torch.npu.memory._dump_snapshot",
        "torch.npu.memory._free_mutex",
        "torch.npu.memory._record_memory_history_impl",
        "torch.npu.memory._set_allocator_settings",
        "torch.npu.memory.empty_cache",
        "torch.npu.mem_get_info",
        "torch.npu.memory.reset_accumulated_host_memory_stats",
        "torch.npu.memory.reset_accumulated_memory_stats",
        "torch.npu.memory.reset_max_memory_allocated",
        "torch.npu.memory.reset_max_memory_cached",
        "torch.npu.memory.reset_peak_host_memory_stats",
        "torch.npu.memory.reset_peak_memory_stats",
        "torch.npu.memory.set_per_process_memory_fraction",
        "torch.npu.random.manual_seed_all",
        "torch.npu.random.manual_seed",
        "torch.npu.random.seed_all",
        "torch.npu.random.seed",
        "torch.npu.set_sync_debug_mode",
        "torch.npu._set_rng_state_offset",
        "torch.npu._get_generator", 
        "torch.npu._memory_viz._frames_fmt", 
        "torch.npu._memory_viz._frame_fmt", 
        "torch.npu.amp.autocast_mode.custom_bwd", 
        "torch.npu.amp.autocast_mode.custom_fwd", 
        "torch.npu.is_initialized"
    ],
    TorchInGraphFunctionVariable,
)

torch_c_binding_in_graph_functions_npu = dict.fromkeys(
    [
        "torch_npu._C._npu_changeCurrentAllocator",
        "torch_npu._C._npu_npuCachingAllocator_set_allocator_settings",
        "torch_npu._C._npu_emptyCache",
        "torch_npu._C._npu_getAllocator",
        "torch_npu._C._npu_getCheckpointState",
        "torch_npu._C._npu_getCurrentStream",
        "torch_npu._C._npu_getDefaultStream",
        "torch_npu._C._npu_init",
        "torch_npu._C._npu_ipc_collect",
        "torch_npu._C._npu_resetAccumulatedHostMemoryStats",
        "torch_npu._C._npu_resetPeakHostMemoryStats",
        "torch_npu._C._npu_resetPeakMemoryStats",
        "torch_npu._C._npu_set_sync_debug_mode",
        "torch_npu._C._npu_setDevice",
        "torch_npu._C._npu_setMemoryFraction",
        "torch_npu._C._npu_synchronize",
    ],
    TorchInGraphFunctionVariable,
)

skip_functions_npu = dict.fromkeys(
    [
        "torch.npu.set_device",
    ],
    SkipFunctionVariable
)


def _patch_npu_trace_rules():
    torch._dynamo.trace_rules.clear_lru_cache()
    torch._dynamo.trace_rules.torch_name_rule_map.append(torch_non_c_binding_in_graph_functions_npu)
    torch._dynamo.trace_rules.torch_name_rule_map.append(torch_c_binding_in_graph_functions_npu)
    torch._dynamo.trace_rules.torch_name_rule_map.append(skip_functions_npu)
    torch_module.constant_fold_functions[torch.npu.current_device] = True
    torch_module.constant_fold_functions[torch.npu.get_device_properties] = True
    torch_module.constant_fold_functions[torch.npu.is_available] = True
    common_constant_types.add(torch_npu._C._NPUDeviceProperties)

