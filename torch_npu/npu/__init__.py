# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "native_device", "npu_device", "is_initialized", "_lazy_call", "_lazy_init", "init", "set_dump",
    "synchronize", "device_count", "can_device_access_peer", "set_device", "current_device", "get_device_name",
    "get_device_properties", "get_device_capability", "_get_device_index", "is_available", "device", "device_of",
    "stream", "current_stream", "default_stream", "set_sync_debug_mode", "get_sync_debug_mode",
    "init_dump", "utilization", "finalize_dump", "set_dump", "manual_seed", "manual_seed_all",
    "seed", "seed_all", "initial_seed", "caching_allocator_alloc", "stress_detect",
    "caching_allocator_delete", "set_per_process_memory_fraction", "empty_cache", "memory_stats",
    "memory_stats_as_nested_dict", "reset_accumulated_memory_stats",
    "reset_peak_memory_stats", "reset_max_memory_allocated", "reset_max_memory_cached",
    "memory_allocated", "max_memory_allocated", "memory_reserved", "max_memory_reserved",
    "memory_cached", "max_memory_cached", "memory_snapshot", "memory_summary",
    "Stream", "mstx", "Event", "profiler", "set_option", "set_aoe", "_in_bad_fork", "set_compile_mode",
    "FloatTensor", "IntTensor", "DoubleTensor", "LongTensor", "ShortTensor",
    "CharTensor", "ByteTensor", "HalfTensor", "set_mm_bmm_format_nd", "get_mm_bmm_format_nd",
    "get_npu_overflow_flag", "clear_npu_overflow_flag", "get_rng_state", "set_rng_state",
    "get_rng_state_all", "set_rng_state_all", "is_jit_compile_false",
    "current_blas_handle", "config", "matmul", "conv", "mem_get_info", "is_bf16_supported", "SyncLaunchStream",
]

from typing import Tuple

import torch
import torch_npu

from .mstx import mstx
from .device import __device__ as native_device
from .device import __npu_device__ as npu_device
from .utils import (is_initialized, _lazy_call, _lazy_init, init, set_dump,
                    synchronize, device_count, can_device_access_peer, set_device, current_device, get_device_name,
                    get_device_properties, get_device_capability, _get_device_index, is_available, device, device_of,
                    stream, current_stream, default_stream, set_sync_debug_mode, get_sync_debug_mode,
                    init_dump, utilization, finalize_dump, set_dump, _in_bad_fork, stress_detect,
                    get_npu_overflow_flag, clear_npu_overflow_flag, current_blas_handle, mem_get_info, is_bf16_supported)
from .random import (manual_seed, manual_seed_all, seed, seed_all, initial_seed,
                     get_rng_state_all, set_rng_state_all,
                     get_rng_state, set_rng_state)
from .memory import (caching_allocator_alloc, caching_allocator_delete,
                     set_per_process_memory_fraction, empty_cache, memory_stats, memory_stats_as_nested_dict,
                     reset_accumulated_memory_stats, reset_peak_memory_stats,
                     reset_max_memory_allocated, reset_max_memory_cached, memory_allocated,
                     max_memory_allocated, memory_reserved, max_memory_reserved,
                     memory_cached, max_memory_cached, memory_snapshot, memory_summary)
from .streams import Stream, Event, SyncLaunchStream
from . import profiler
from .npu_frontend_enhance import (set_option, set_aoe, set_compile_mode, set_mm_bmm_format_nd, get_mm_bmm_format_nd,
                                   is_jit_compile_false)
from .backends import *

torch.optim.Optimizer._hook_for_profile = profiler._hook_for_profile
config = npu_frontend_enhance.npuConfig()

matmul = npu_frontend_enhance.allowHF32Matmul()
conv = npu_frontend_enhance.allowHF32Conv()

default_generators: Tuple[torch._C.Generator] = ()
