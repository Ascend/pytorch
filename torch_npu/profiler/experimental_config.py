from typing import Union

import torch_npu._C

from .analysis.prof_common_func._constant import Constant, print_warn_msg, print_info_msg
from .analysis.prof_common_func._cann_package_manager import CannPackageManager

__all__ = [
    "_ExperimentalConfig", 
    "supported_profiler_level", 
    "supported_ai_core_metrics",
    "supported_export_type",
    "ProfilerLevel", 
    "AiCMetrics",
    "ExportType",
    "HostSystem"
]


def supported_profiler_level():
    return set((ProfilerLevel.Level0, ProfilerLevel.Level1, ProfilerLevel.Level2, ProfilerLevel.Level_none))


def supported_ai_core_metrics():
    return set((AiCMetrics.AiCoreNone, AiCMetrics.PipeUtilization, AiCMetrics.ArithmeticUtilization,
                AiCMetrics.Memory, AiCMetrics.MemoryL0, AiCMetrics.MemoryUB,
                AiCMetrics.ResourceConflictRatio, AiCMetrics.L2Cache, AiCMetrics.MemoryAccess))


def supported_export_type():
    return set((ExportType.Db, ExportType.Text))


class ProfilerLevel:
    Level0 = Constant.LEVEL0
    Level1 = Constant.LEVEL1
    Level2 = Constant.LEVEL2
    Level_none = Constant.LEVEL_NONE


class AiCMetrics:
    PipeUtilization = Constant.AicPipeUtilization
    ArithmeticUtilization = Constant.AicArithmeticUtilization
    Memory = Constant.AicMemory
    MemoryL0 = Constant.AicMemoryL0
    MemoryUB = Constant.AicMemoryUB
    ResourceConflictRatio = Constant.AicResourceConflictRatio
    L2Cache = Constant.AicL2Cache
    MemoryAccess = Constant.AicMemoryAccess
    AiCoreNone = Constant.AicMetricsNone


class ExportType:
    Db = Constant.Db
    Text = Constant.Text


class HostSystem:
    CPU = Constant.CPU
    MEM = Constant.MEM
    DISK = Constant.DISK
    NETWORK = Constant.NETWORK
    OSRT = Constant.OSRT


class _ExperimentalConfig:
    def __init__(self,
                 profiler_level: int = Constant.LEVEL0,
                 aic_metrics: int = Constant.AicMetricsNone,
                 l2_cache: bool = False,
                 msprof_tx: bool = False,
                 data_simplification: bool = True,
                 record_op_args: bool = False,
                 op_attr: bool = False,
                 gc_detect_threshold: float = None,
                 export_type: Union[str, list] = None,
                 host_sys: list = None,
                 mstx_domain_include: list = None,
                 mstx_domain_exclude: list = None,
                 sys_io: bool = False,
                 sys_interconnection: bool = False):
        self._profiler_level = profiler_level
        self._aic_metrics = aic_metrics
        if self._profiler_level != Constant.LEVEL_NONE:
            if self._profiler_level != Constant.LEVEL0 and self._aic_metrics == Constant.AicMetricsNone:
                self._aic_metrics = Constant.AicPipeUtilization
        self._l2_cache = l2_cache
        self._msprof_tx = msprof_tx
        self._data_simplification = data_simplification
        self.record_op_args = record_op_args
        self._export_type = self._conver_export_type_to_list(export_type)
        self._host_sys = host_sys if host_sys else []
        self._op_attr = op_attr
        self._gc_detect_threshold = gc_detect_threshold
        self._mstx_domain_include = mstx_domain_include if mstx_domain_include else []
        self._mstx_domain_exclude = mstx_domain_exclude if mstx_domain_exclude else []
        self._sys_io = sys_io
        self._sys_interconnection = sys_interconnection
        self._check_params()
        self._check_mstx_domain_params()
        self._check_host_sys_params()

    def __call__(self) -> torch_npu._C._profiler._ExperimentalConfig:
        return torch_npu._C._profiler._ExperimentalConfig(trace_level=self._profiler_level,
                                                          metrics=self._aic_metrics,
                                                          l2_cache=self._l2_cache,
                                                          record_op_args=self.record_op_args,
                                                          msprof_tx=self._msprof_tx,
                                                          op_attr=self._op_attr,
                                                          host_sys=self._host_sys,
                                                          mstx_domain_include=self._mstx_domain_include,
                                                          mstx_domain_exclude=self._mstx_domain_exclude,
                                                          sys_io=self._sys_io,
                                                          sys_interconnection=self._sys_interconnection)

    @property
    def export_type(self):
        return self._export_type

    @property
    def with_gc(self):
        return self._gc_detect_threshold is not None

    @property
    def gc_detect_threshold(self):
        return self._gc_detect_threshold

    def _conver_export_type_to_list(self, export_type: Union[str, list]) -> list:
        if not export_type:
            print_warn_msg(f"Invalid parameter export_type: {export_type}, reset it to text.")
            return [ExportType.Text]
        if isinstance(export_type, str):
            return [export_type]
        elif isinstance(export_type, list):
            try:
                return list(set(export_type))
            except Exception as error:
                print_warn_msg(f"Invalid parameter export_type: {export_type}, reset it to text. Error is {error}")
                return [ExportType.Text]
        else:
            print_warn_msg(f"Invalid parameter export_type: {export_type}, reset it to text.")
            return [ExportType.Text]

    def _check_params(self):
        if (self._profiler_level == Constant.LEVEL0 or self._profiler_level == Constant.LEVEL_NONE) and \
            self._aic_metrics != Constant.AicMetricsNone:
            print_warn_msg("Please use level1 or level2 if you want to collect aic metrics, reset aic metrics to None!")
            self._aic_metrics = Constant.AicMetricsNone
        if not isinstance(self._l2_cache, bool):
            print_warn_msg("Invalid parameter l2_cache, which must be of boolean type, reset it to False.")
            self._l2_cache = False
        if not isinstance(self._msprof_tx, bool):
            print_warn_msg("Invalid parameter msprof_tx, which must be of boolean type, reset it to False.")
            self._msprof_tx = False
        if self._data_simplification is not None and not isinstance(self._data_simplification, bool):
            print_warn_msg("Invalid parameter data_simplification, which must be of boolean type, reset it to default.")
            self._data_simplification = True
        if not isinstance(self.record_op_args, bool):
            print_warn_msg("Invalid parameter record_op_args, which must be of boolean type, reset it to False.")
            self.record_op_args = False
        if self._profiler_level not in \
           (ProfilerLevel.Level0, ProfilerLevel.Level1, ProfilerLevel.Level2, ProfilerLevel.Level_none):
            print_warn_msg("Invalid parameter profiler_level, reset it to ProfilerLevel.Level0.")
            self._profiler_level = ProfilerLevel.Level0
        if self._aic_metrics not in (
                AiCMetrics.L2Cache, AiCMetrics.MemoryL0, AiCMetrics.Memory, AiCMetrics.MemoryUB,
                AiCMetrics.PipeUtilization, AiCMetrics.ArithmeticUtilization, AiCMetrics.ResourceConflictRatio,
                AiCMetrics.MemoryAccess, AiCMetrics.AiCoreNone):
            print_warn_msg("Invalid parameter aic_metrics, reset it to default.")
            if self._profiler_level == ProfilerLevel.Level0:
                self._aic_metrics = AiCMetrics.AiCoreNone
            else:
                self._aic_metrics = AiCMetrics.PipeUtilization
        if not isinstance(self._op_attr, bool):
            print_warn_msg("Invalid parameter op_attr, which must be of boolean type, reset it to False.")
            self._op_attr = False
        if not all(export_type in [ExportType.Text, ExportType.Db] for export_type in self._export_type):
            print_warn_msg("Invalid parameter export_type, reset it to text.")
            self._export_type = [ExportType.Text]
        if self._op_attr and ExportType.Db not in self._export_type:
            print_warn_msg("op_attr switch is invalid with export type set as text.")
            self._op_attr = False
        if self._gc_detect_threshold is not None:
            if not isinstance(self._gc_detect_threshold, (int, float)):
                print_warn_msg("Parameter gc_detect_threshold is not int or float type, reset it to default.")
                self._gc_detect_threshold = None
            elif self._gc_detect_threshold < 0.0:
                print_warn_msg("Parameter gc_detect_threshold can not be negative, reset it to default.")
                self._gc_detect_threshold = None
            elif self._gc_detect_threshold == 0.0:
                print_info_msg("Parameter gc_detect_threshold is set to 0, it will collect all gc events.")
        if not isinstance(self._sys_io, bool):
            print_warn_msg("Invalid parameter sys_io, which must be of boolean type, reset it to False.")
            self._sys_io = False
        if not isinstance(self._sys_interconnection, bool):
            print_warn_msg("Invalid parameter sys_interconnection, which must be of boolean type, reset it to False.")
            self._sys_interconnection = False

    def _check_mstx_domain_params(self):
        if not self._msprof_tx:
            if self._mstx_domain_include or self._mstx_domain_exclude:
                print_warn_msg("mstx_domain_include and mstx_domain_exclude are only valid when msprof_tx is True.")
            self._mstx_domain_include = []
            self._mstx_domain_exclude = []
            return
        if self._mstx_domain_include:
            if not isinstance(self._mstx_domain_include, list):
                print_warn_msg("Invalid parameter mstx_domain_include, which must be of list type, " \
                               "reset it to default.")
                self._mstx_domain_include = []
            if any(not isinstance(domain, str) for domain in self._mstx_domain_include):
                print_warn_msg("Invalid parameter mstx_domain_include, which contents must be of str type, " \
                               "reset it to default.")
                self._mstx_domain_include = []
            else:
                self._mstx_domain_include = list(set(self._mstx_domain_include))
        if self._mstx_domain_exclude:
            if not isinstance(self._mstx_domain_exclude, list):
                print_warn_msg("Invalid parameter mstx_domain_exclude, which must be of list type, " \
                               "reset it to default.")
                self._mstx_domain_exclude = []
            if any(not isinstance(domain, str) for domain in self._mstx_domain_exclude):
                print_warn_msg("Invalid parameter _mstx_domain_exclude, which contents must be of str type, " \
                               "reset it to default.")
                self._mstx_domain_exclude = []
            else:
                self._mstx_domain_exclude = list(set(self._mstx_domain_exclude))
        if self._mstx_domain_include and self._mstx_domain_exclude:
            print_warn_msg("Parameter mstx_domain_include and mstx_domain_exclude can not be both set, " \
                           "only mstx_domain_include will work.")
            self._mstx_domain_exclude = []

    def _check_host_sys_params(self):
        if not isinstance(self._host_sys, list):
            print_warn_msg("Invalid parameter host_sys, which must be of list type, reset it to empty.")
            self._host_sys = []
        if not all(host_sys in [HostSystem.CPU, HostSystem.MEM, HostSystem.DISK, HostSystem.NETWORK, HostSystem.OSRT]
                   for host_sys in self._host_sys):
            print_warn_msg("Invalid parameter host_sys, reset it to empty.")
            self._host_sys = []
        self._host_sys = list(set(str(item) for item in self._host_sys))