from torch_npu._C._profiler import ProfilerActivity

from .profiler import (
    profile,
    _KinetoProfile,
    tensorboard_trace_handler)
from .profiler_interface import supported_activities
from .scheduler import Schedule as schedule
from .scheduler import ProfilerAction
from .experimental_config import _ExperimentalConfig, supported_profiler_level, supported_ai_core_metrics, \
    supported_export_type, ProfilerLevel, AiCMetrics, ExportType, HostSystem
from ._non_intrusive_profile import _NonIntrusiveProfile

__all__ = ["profile", "ProfilerActivity", "supported_activities", "tensorboard_trace_handler", "schedule",
           "ProfilerAction", "_ExperimentalConfig", "supported_profiler_level", "supported_ai_core_metrics",
           "supported_export_type", "ProfilerLevel", "AiCMetrics", "ExportType", "HostSystem"]


_NonIntrusiveProfile.init()
