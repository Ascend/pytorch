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
    "synchronize", "device_count", "set_device", "current_device", "get_device_name",
    "get_device_properties", "_get_device_index", "is_available", "device", "device_of",
    "stream", "current_stream", "default_stream", "init_dump",
    "finalize_dump", "set_dump", "manual_seed", "manual_seed_all",
    "seed", "seed_all", "initial_seed", "_free_mutex", "caching_allocator_alloc",
    "caching_allocator_delete", "empty_cache", "memory_stats",
    "memory_stats_as_nested_dict", "reset_accumulated_memory_stats",
    "reset_peak_memory_stats", "reset_max_memory_allocated", "reset_max_memory_cached",
    "memory_allocated", "max_memory_allocated", "memory_reserved", "max_memory_reserved",
    "memory_cached", "max_memory_cached", "memory_snapshot", "memory_summary",
    "Stream", "Event", "profiler", "set_option", "set_aoe", "profile", "prof_init",
    "prof_start", "prof_stop", "prof_finalize", "iteration_start", "iteration_end",
    "profileConfig", "_in_bad_fork", "set_compile_mode",
    "FloatTensor", "IntTensor", "DoubleTensor", "LongTensor", "ShortTensor", 
    "CharTensor", "ByteTensor", "HalfTensor", "set_mm_bmm_format_nd", "get_mm_bmm_format_nd",
    "get_npu_overflow_flag", "clear_npu_overflow_flag", "get_rng_state", "set_rng_state",
    "get_rng_state_all", "set_rng_state_all", "make_replay_graph", "is_jit_compile_false"
]

import torch

from .device import __device__ as native_device
from .device import __npu_device__ as npu_device
from .utils import (is_initialized, _lazy_call, _lazy_init, init, set_dump,
                    synchronize, device_count, set_device, current_device, get_device_name,
                    get_device_properties, _get_device_index, is_available, device, device_of,
                    stream, current_stream, default_stream, init_dump,
                    finalize_dump, set_dump, _in_bad_fork, get_npu_overflow_flag,
                    clear_npu_overflow_flag)
from .random import (manual_seed, manual_seed_all, seed, seed_all, initial_seed, 
                     get_rng_state_all, set_rng_state_all,
                     get_rng_state, set_rng_state)
from .memory import (_free_mutex, caching_allocator_alloc, caching_allocator_delete,
                     empty_cache, memory_stats, memory_stats_as_nested_dict,
                     reset_accumulated_memory_stats, reset_peak_memory_stats,
                     reset_max_memory_allocated, reset_max_memory_cached, memory_allocated,
                     max_memory_allocated, memory_reserved, max_memory_reserved,
                     memory_cached, max_memory_cached, memory_snapshot, memory_summary)
from .streams import Stream, Event
from .graph import is_graph_mode, disable_graph_mode, enable_graph_mode, launch_graph
from .replay_graph import make_replay_graph
from . import profiler
from .npu_frontend_enhance import (set_option, set_aoe, profile, prof_init,
            prof_start, prof_stop, prof_finalize, iteration_start, iteration_end, 
            profileConfig, set_compile_mode, set_mm_bmm_format_nd, get_mm_bmm_format_nd, is_jit_compile_false)
from .tensor import FloatTensor, IntTensor, DoubleTensor, LongTensor, ShortTensor, CharTensor, ByteTensor, HalfTensor

torch.optim.Optimizer._hook_for_profile = profiler._hook_for_profile