from enum import Enum

import torch_npu._C

from .analysis.prof_common_func.constant import Constant


def supported_profiler_level():
    return set(ProfilerLevel.__members__.values())


def supported_ai_core_metrics():
    return set(AiCMetrics.__members__.values())


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
                 record_op_args: bool = False):
        self._profiler_level = profiler_level
        self._aic_metrics = aic_metrics
        if self._profiler_level != Constant.LEVEL0 and self._aic_metrics == Constant.AicMetricsNone:
            self._aic_metrics = Constant.AicPipeUtilization
        self._l2_cache = l2_cache
        self.record_op_args = record_op_args

    def __call__(self) -> torch_npu._C._profiler._ExperimentalConfig:
        return torch_npu._C._profiler._ExperimentalConfig(trace_level=self._profiler_level,
                                                          metrics=self._aic_metrics,
                                                          l2_cache=self._l2_cache,
                                                          record_op_args=self.record_op_args)

    def profiler_level(self):
        return self._profiler_level

    def aic_metrics(self):
        return self._aic_metrics

    def l2_cache(self):
        return self._l2_cache
