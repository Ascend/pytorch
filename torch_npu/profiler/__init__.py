# Copyright (c) 2023, Huawei Technologies.
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

from torch_npu._C._profiler import ProfilerActivity

from .profiler import (
    profile,
    _KinetoProfile,
    supported_activities,
    tensorboard_trace_handler)
from .scheduler import Schedule as schedule
from .scheduler import ProfilerAction
from .experimental_config import _ExperimentalConfig, supported_profiler_level, supported_ai_core_metrics, \
    supported_export_type, ProfilerLevel, AiCMetrics, ExportType
from ._non_intrusive_profile import _NonIntrusiveProfile

__all__ = ["profile", "ProfilerActivity", "supported_activities", "tensorboard_trace_handler", "schedule",
           "ProfilerAction", "_ExperimentalConfig", "supported_profiler_level", "supported_ai_core_metrics",
           "supported_export_type", "ProfilerLevel", "AiCMetrics", "ExportType"]


_NonIntrusiveProfile.init()
