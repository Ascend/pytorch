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
    "is_initialized", "_lazy_call", "_lazy_init", "init", "set_dump",
    "synchronize", "device_count", "set_device", "current_device", "get_device_name",
    "get_device_properties", "get_device_capability", "_get_device_index", "is_available", "device", "device_of",
    "stream", "set_stream", "current_stream", "default_stream", "init_dump",
    "finalize_dump", "set_dump", "manual_seed", "manual_seed_all",
    "seed", "seed_all", "initial_seed", "_free_mutex", "caching_allocator_alloc",
    "caching_allocator_delete", "set_per_process_memory_fraction", "empty_cache", "memory_stats",
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
    "get_rng_state_all", "set_rng_state_all", "make_replay_graph", "is_jit_compile_false",
    "dump_enable", "dump_disable", "get_amp_supported_dtype", "is_autocast_enabled", "set_autocast_enabled",
    "get_autocast_dtype", "set_autocast_dtype"
]

from typing import Tuple, Union
import torch
import torch_npu

from .utils import (is_initialized, _lazy_call, _lazy_init, init, set_dump,
                    synchronize, device_count, set_device, current_device, get_device_name,
                    get_device_properties, get_device_capability, _get_device_index, is_available, device, device_of,
                    stream, set_stream, current_stream, default_stream, init_dump,
                    finalize_dump, set_dump, _in_bad_fork, get_npu_overflow_flag,
                    clear_npu_overflow_flag)
from .random import *  # noqa: F403
from .memory import *  # noqa: F403
from .streams import Stream, Event
from .graph import is_graph_mode, disable_graph_mode, enable_graph_mode, launch_graph
from .replay_graph import make_replay_graph
from . import profiler
from .npu_config import *  # noqa: F403
from .datadump import dump_enable, dump_disable
from .autocast_utils import *  # noqa: F403

torch.optim.Optimizer._hook_for_profile = profiler._hook_for_profile
config = npu_config.npuConfig()

matmul = npu_config.allowHF32Matmul()
conv = npu_config.allowHF32Conv()

# TODO: _npu_isInBadFork is not yet implemented
_is_in_bad_fork = getattr(torch_npu._C, "_npu_isInBadFork", lambda: False)
default_generators: Tuple[torch._C.Generator] = ()  # type: ignore[assignment]

def _get_device(device: Union[int, str, torch.device]) -> torch.device:
    r"""Return the torch.device type object from the passed in device.

    Args:
        device (torch.device or int): selected device.
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device('npu', device)
    return device

def _get_generator(device: torch.device) -> torch._C.Generator:
    r"""Return the NPU Generator object for the given device.

    Args:
        device (torch.device): selected device.
    """

    idx = device.index
    if idx is None:
        idx = current_device()
    return torch.npu.default_generators[idx]

def _set_rng_state_offset(offset: int, device: Union[int, str, torch.device] = 'npu') -> None:
    r"""Sets the random number generator state offset of the specified NPU.

    Args:
        offset (int): The desired offset
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'npu'`` (i.e., ``torch.device('npu')``, the current NPU device).
    """
    final_device = _get_device(device)

    def cb():
        default_generator = _get_generator(final_device)
        default_generator.set_offset(offset)

    _lazy_call(cb)

def _get_rng_state_offset(device: Union[int, str, torch.device] = 'npu') -> int:
    r"""Returns the random number generator state offset of the specified NPU.

    Args:
        device (torch.device or int, optional): The device to return the RNG state offset of.
            Default: ``'npu'`` (i.e., ``torch.device('npu')``, the current NPU device).

    .. warning::
        This function eagerly initializes NPU.
    """
    _lazy_init()
    final_device = _get_device(device)
    default_generator = _get_generator(final_device)
    return default_generator.get_offset()
