import torch_npu._C

from .analysis.prof_common_func.constant import Constant, print_warn_msg


def supported_profiler_level():
    return set((ProfilerLevel.Level0, ProfilerLevel.Level1, ProfilerLevel.Level2))


def supported_ai_core_metrics():
    return set((AiCMetrics.PipeUtilization, AiCMetrics.ArithmeticUtilization,
                AiCMetrics.Memory, AiCMetrics.MemoryL0, AiCMetrics.MemoryUB,
                AiCMetrics.ResourceConflictRatio, AiCMetrics.L2Cache))


class ProfilerLevel:
    Level0 = Constant.LEVEL0
    Level1 = Constant.LEVEL1
    Level2 = Constant.LEVEL2


class AiCMetrics:
    PipeUtilization = Constant.AicPipeUtilization
    ArithmeticUtilization = Constant.AicArithmeticUtilization
    Memory = Constant.AicMemory
    MemoryL0 = Constant.AicMemoryL0
    MemoryUB = Constant.AicMemoryUB
    ResourceConflictRatio = Constant.AicResourceConflictRatio
    L2Cache = Constant.AicL2Cache


class _ExperimentalConfig:
    def __init__(self,
                 profiler_level: int = Constant.LEVEL0,
                 aic_metrics: int = Constant.AicMetricsNone,
                 l2_cache: bool = False,
                 data_simplification: bool = True,
                 record_op_args: bool = False):
        self._profiler_level = profiler_level
        self._aic_metrics = aic_metrics
        if self._profiler_level != Constant.LEVEL0 and self._aic_metrics == Constant.AicMetricsNone:
            self._aic_metrics = Constant.AicPipeUtilization
        self._l2_cache = l2_cache
        self._data_simplification = data_simplification
        self.record_op_args = record_op_args
        self._check_params()

    def __call__(self) -> torch_npu._C._profiler._ExperimentalConfig:
        return torch_npu._C._profiler._ExperimentalConfig(trace_level=self._profiler_level,
                                                          metrics=self._aic_metrics,
                                                          l2_cache=self._l2_cache,
                                                          record_op_args=self.record_op_args)

    def _check_params(self):
        if self._profiler_level == Constant.LEVEL0 and self._aic_metrics != Constant.AicMetricsNone:
            print_warn_msg("Please use leve1 or level2 if you want to collect aic metrics!")
        if not isinstance(self._l2_cache, bool):
            print_warn_msg("Invalid parameter l2_cache, which must be of boolean type, reset it to False.")
            self._l2_cache = False
        if self._data_simplification is not None and not isinstance(self._data_simplification, bool):
            print_warn_msg("Invalid parameter data_simplification, which must be of boolean type, reset it to default.")
            self._data_simplification = None
        if not isinstance(self.record_op_args, bool):
            print_warn_msg("Invalid parameter record_op_args, which must be of boolean type, reset it to False.")
            self.record_op_args = False
        if self._profiler_level not in (ProfilerLevel.Level0, ProfilerLevel.Level1, ProfilerLevel.Level2):
            print_warn_msg("Invalid parameter profiler_level, reset it to ProfilerLevel.Level0.")
            self._profiler_level = ProfilerLevel.Level0
        if self._aic_metrics not in (
                AiCMetrics.L2Cache, AiCMetrics.MemoryL0, AiCMetrics.Memory, AiCMetrics.MemoryUB,
                AiCMetrics.PipeUtilization, AiCMetrics.ArithmeticUtilization, AiCMetrics.ResourceConflictRatio,
                Constant.AicMetricsNone):
            print_warn_msg("Invalid parameter aic_metrics, reset it to default.")
            if self._profiler_level == ProfilerLevel.Level0:
                self._aic_metrics = Constant.AicMetricsNone
            else:
                self._aic_metrics = AiCMetrics.PipeUtilization
