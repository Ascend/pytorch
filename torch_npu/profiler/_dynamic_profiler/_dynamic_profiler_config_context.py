import json
from torch_npu._C._profiler import ProfilerActivity
from ..experimental_config import _ExperimentalConfig, ProfilerLevel, AiCMetrics
from ._dynamic_profiler_utils import DynamicProfilerUtils


class ConfigContext:
    DEFAULT_ACTIVE_NUM = 1
    DEFAULT_START_STEP = 0
    INSTANT_START_STEP = -1
    DEFAULT_WARMUP = 0
    DEADLINE_PROF_DIR = "./"
    BOOL_MAP = {'true': True, 'false': False}

    def __init__(self, json_data: dict):
        self.activity_set = set()
        self.prof_path = str()
        self._analyse = False
        self.record_shapes = False
        self.profile_memory = False
        self.with_stack = False
        self.with_flops = False
        self.with_modules = False
        self.is_rank = False
        self._rank_set = set()
        self.experimental_config = None
        self._active = 1
        self._start_step = 0
        self._warmup = 0
        self.is_valid = False
        self._meta_data = {}
        self._async_mode = False
        self._is_dyno = DynamicProfilerUtils.is_dyno_model()
        self._is_dyno_monitor = False
        self._rank_id = DynamicProfilerUtils.get_rank_id()
        self.parse(json_data)

    def parse(self, json_data: dict):
        self.is_valid = json_data.get("is_valid", False)
        self._is_dyno_monitor = "NPU_MONITOR_START" in json_data
        if self._is_dyno_monitor:
            return
        self._parse_activity(json_data)
        self._parse_prof_dir(json_data)
        self._meta_data = json_data.get('metadata', {})
        self._parse_analysis(json_data)
        self._parse_async_mode(json_data)
        self._parse_report_shape(json_data)
        self._parse_profiler_memory(json_data)
        self._parse_with_flops(json_data)
        self._parse_with_stack(json_data)
        self._parse_with_modules(json_data)
        self._parse_active(json_data)
        self._parse_warmup(json_data)
        self._parse_start_step(json_data)
        self._parse_exp_cfg(json_data)
        self._parse_ranks(json_data)

    def _parse_start_step(self, json_data: dict):
        if not self._is_dyno:
            self._start_step = json_data.get("start_step", self.DEFAULT_START_STEP)
        else:
            start_step = json_data.get("PROFILE_START_STEP", self.DEFAULT_START_STEP)
            try:
                self._start_step = int(start_step)
            except ValueError:
                self._start_step = self.DEFAULT_START_STEP

        if not isinstance(self._start_step, int) or self._start_step < self.INSTANT_START_STEP:
            DynamicProfilerUtils.out_log("Start step is not valid, will be reset to {}.".format(
                self.DEFAULT_START_STEP), DynamicProfilerUtils.LoggerLevelEnum.INFO)
            self._start_step = self.DEFAULT_START_STEP
        DynamicProfilerUtils.out_log("Start step will be set to {}.".format(
            self._start_step), DynamicProfilerUtils.LoggerLevelEnum.INFO)

    def _parse_prof_dir(self, json_data: dict):
        if not self._is_dyno:
            self.prof_path = json_data.get('prof_dir', self.DEADLINE_PROF_DIR)
        else:
            self.prof_path = json_data.get("ACTIVITIES_LOG_FILE", self.DEADLINE_PROF_DIR)

    def _parse_active(self, json_data: dict):
        if not self._is_dyno:
            self._active = json_data.get("active", self.DEFAULT_ACTIVE_NUM)
        else:
            active = json_data.get("ACTIVITIES_ITERATIONS", self.DEFAULT_ACTIVE_NUM)
            try:
                self._active = int(active)
            except ValueError:
                self._active = self.DEFAULT_ACTIVE_NUM

    def _parse_warmup(self, json_data: dict):
        if not self._is_dyno:
            self._warmup = json_data.get("warmup", self.DEFAULT_WARMUP)
        else:
            warmup = json_data.get("WARMUP_ITERATIONS", self.DEFAULT_WARMUP)
            try:
                self._warmup = int(warmup)
            except ValueError:
                self._warmup = self.DEFAULT_WARMUP

    def _parse_with_stack(self, json_data: dict):
        if not self._is_dyno:
            self.with_stack = json_data.get('with_stack', False)
        else:
            with_stack = json_data.get("PROFILE_WITH_STACK")
            if isinstance(with_stack, str):
                self.with_stack = self.BOOL_MAP.get(with_stack.lower(), False)
            else:
                self.with_stack = False

    def _parse_report_shape(self, json_data: dict):
        if not self._is_dyno:
            self.record_shapes = json_data.get('record_shapes', False)
        else:
            record_shapes = json_data.get("PROFILE_RECORD_SHAPES")
            if isinstance(record_shapes, str):
                self.record_shapes = self.BOOL_MAP.get(record_shapes.lower(), False)
            else:
                self.record_shapes = False

    def _parse_profiler_memory(self, json_data: dict):
        if not self._is_dyno:
            self.profile_memory = json_data.get('profile_memory', None)
        else:
            profile_memory = json_data.get("PROFILE_PROFILE_MEMORY")
            if isinstance(profile_memory, str):
                self.profile_memory = self.BOOL_MAP.get(profile_memory.lower(), False)
            else:
                self.profile_memory = False

    def _parse_with_flops(self, json_data: dict):
        if not self._is_dyno:
            self.with_flops = json_data.get('with_flops', False)
        else:
            with_flops = json_data.get("PROFILE_WITH_FLOPS")
            if isinstance(with_flops, str):
                self.with_flops = self.BOOL_MAP.get(with_flops.lower(), False)
            else:
                self.with_flops = False

    def _parse_with_modules(self, json_data: dict):
        if not self._is_dyno:
            self.with_modules = json_data.get('with_modules', False)
        else:
            with_modules = json_data.get("PROFILE_WITH_MODULES")
            if isinstance(with_modules, str):
                self.with_modules = self.BOOL_MAP.get(with_modules.lower(), False)
            else:
                self.with_modules = False

    def _parse_ranks(self, json_data: dict):
        self.is_rank = json_data.get("is_rank", False)
        if not isinstance(self.is_rank, bool):
            self.is_rank = False
            DynamicProfilerUtils.out_log("Set is_rank failed, is_rank must be bool!",
                                         DynamicProfilerUtils.LoggerLevelEnum.WARNING)
            return
        if not self.is_rank:
            return
        DynamicProfilerUtils.out_log("Set is_rank success!", DynamicProfilerUtils.LoggerLevelEnum.INFO)
        ranks = json_data.get("rank_list", False)
        if not isinstance(ranks, list):
            DynamicProfilerUtils.out_log("Set rank_list failed, rank_list must be list!",
                                         DynamicProfilerUtils.LoggerLevelEnum.WARNING)
            return
        for rank in ranks:
            if isinstance(rank, int) and rank >= 0:
                self._rank_set.add(rank)

    def _parse_activity(self, json_data: dict):
        if not self._is_dyno:
            activities = json_data.get('activities')
        else:
            activities = json_data.get('PROFILE_ACTIVITIES').split(",")
        if activities and isinstance(activities, list):
            for entry in activities:
                activity = getattr(ProfilerActivity, entry.upper(), None)
                if activity:
                    self.activity_set.add(activity)
                else:
                    DynamicProfilerUtils.out_log("Set activity failed, activity must be CPU OR NPU!",
                                                 DynamicProfilerUtils.LoggerLevelEnum.WARNING)

    def _parse_analysis(self, json_data: dict):
        if not self._is_dyno:
            self._analyse = json_data.get("analyse", False)
        else:
            self._analyse = json_data.get("PROFILE_ANALYSE", 'false')
            self._analyse = self.BOOL_MAP.get(self._analyse.lower(), False)

    def _parse_async_mode(self, json_data: dict):
        if not self._is_dyno:
            self._async_mode = json_data.get('async_mode', False)
        else:
            self._async_mode = json_data.get("PROFILE_ASYNC_MODE", 'false')
            self._async_mode = self.BOOL_MAP.get(self._async_mode.lower(), False)
        if not self._analyse and self._async_mode:
            DynamicProfilerUtils.out_log("When analyse is False, async_mode will not take effect!",
                                         DynamicProfilerUtils.LoggerLevelEnum.WARNING)

    def _parse_dyno_exp_cfg(self, json_data: dict):
        profiler_level = json_data.get('PROFILE_PROFILER_LEVEL', 'Level0')
        profiler_level = getattr(ProfilerLevel, profiler_level, profiler_level)
        aic_metrics = json_data.get('PROFILE_AIC_METRICS', 'AiCoreNone')
        aic_metrics = getattr(AiCMetrics, aic_metrics, aic_metrics)
        l2_cache = json_data.get('PROFILE_L2_CACHE', 'false')
        l2_cache = self.BOOL_MAP.get(l2_cache.lower(), False)
        op_attr = json_data.get('PROFILE_OP_ATTR', 'false')
        op_attr = self.BOOL_MAP.get(op_attr.lower(), False)
        gc_detect_threshold = json_data.get('PROFILE_GC_DETECT_THRESHOLD', None)
        if isinstance(gc_detect_threshold, str):
            gc_detect_threshold = None if gc_detect_threshold == "None" else float(gc_detect_threshold)
        data_simplification = json_data.get('PROFILE_DATA_SIMPLIFICATION', 'true')
        data_simplification = self.BOOL_MAP.get(data_simplification.lower(), True)
        record_op_args = False
        export_type = json_data.get('PROFILE_EXPORT_TYPE', 'text').lower()
        msprof_tx = json_data.get('PROFILE_MSPROF_TX', 'false')
        msprof_tx = self.BOOL_MAP.get(msprof_tx.lower(), False)
        mstx = json_data.get('PROFILE_MSTX', 'false')
        mstx = self.BOOL_MAP.get(mstx.lower(), False)
        host_sys = DynamicProfilerUtils.parse_str_params_to_list(json_data.get('PROFILE_HOST_SYS', None))
        mstx_domain_include = DynamicProfilerUtils.parse_str_params_to_list(json_data.get('PROFILE_MSTX_DOMAIN_INCLUDE', None))
        mstx_domain_exclude = DynamicProfilerUtils.parse_str_params_to_list(json_data.get('PROFILE_MSTX_DOMAIN_EXCLUDE', None))
        sys_io = json_data.get('PROFILE_SYS_IO', 'false')
        sys_io = self.BOOL_MAP.get(sys_io.lower(), False)
        sys_interconnection = json_data.get('PROFILE_SYS_INTERCONNECTION', 'false')
        sys_interconnection = self.BOOL_MAP.get(sys_interconnection.lower(), False)

        self.experimental_config = _ExperimentalConfig(
            profiler_level=profiler_level,
            aic_metrics=aic_metrics,
            l2_cache=l2_cache,
            op_attr=op_attr,
            gc_detect_threshold=gc_detect_threshold,
            data_simplification=data_simplification,
            record_op_args=record_op_args,
            export_type=export_type,
            msprof_tx=msprof_tx,
            mstx=mstx,
            host_sys=host_sys,
            mstx_domain_include=mstx_domain_include,
            mstx_domain_exclude=mstx_domain_exclude,
            sys_io=sys_io,
            sys_interconnection=sys_interconnection,
        )

    def _parse_cfg_json_exp_cfg(self, json_data: dict):
        exp_config = json_data.get('experimental_config')
        if not exp_config:
            self.experimental_config = None
            return
        profiler_level = exp_config.get('profiler_level', 'Level0')
        profiler_level = getattr(ProfilerLevel, profiler_level, profiler_level)
        aic_metrics = exp_config.get('aic_metrics', 'AiCoreNone')
        aic_metrics = getattr(AiCMetrics, aic_metrics, aic_metrics)
        l2_cache = exp_config.get('l2_cache', False)
        op_attr = exp_config.get('op_attr', False)
        gc_detect_threshold = exp_config.get('gc_detect_threshold', None)
        data_simplification = exp_config.get('data_simplification', True)
        record_op_args = exp_config.get('record_op_args', False)
        export_type = exp_config.get('export_type', 'text')
        msprof_tx = exp_config.get('msprof_tx', False)
        mstx = exp_config.get('mstx', False)
        mstx_domain_include = exp_config.get('mstx_domain_include', None)
        mstx_domain_exclude = exp_config.get('mstx_domain_exclude', None)
        host_sys = exp_config.get('host_sys', None)
        sys_io = exp_config.get('sys_io', None)
        sys_interconnection = exp_config.get('sys_interconnection', None)

        self.experimental_config = _ExperimentalConfig(
            profiler_level=profiler_level,
            aic_metrics=aic_metrics,
            l2_cache=l2_cache,
            op_attr=op_attr,
            gc_detect_threshold=gc_detect_threshold,
            data_simplification=data_simplification,
            record_op_args=record_op_args,
            export_type=export_type,
            msprof_tx=msprof_tx,
            mstx=mstx,
            mstx_domain_include=mstx_domain_include,
            mstx_domain_exclude=mstx_domain_exclude,
            host_sys=host_sys,
            sys_io=sys_io,
            sys_interconnection=sys_interconnection
        )

    def _parse_exp_cfg(self, json_data: dict):
        if not self._is_dyno:
            self._parse_cfg_json_exp_cfg(json_data)
        else:
            self._parse_dyno_exp_cfg(json_data)

    def valid(self) -> bool:
        if not self.is_valid:
            return False
        if not self.is_rank:
            return True
        if self._rank_id in self._rank_set:
            if not self._async_mode:
                self._analyse = False
                DynamicProfilerUtils.out_log("Rank {} is in rank_list {} and async_mode is false, "
                                             "profiler data analyse will be closed !".format(self._rank_id,
                                                                                             self._rank_set),
                                             DynamicProfilerUtils.LoggerLevelEnum.INFO)
            return True
        DynamicProfilerUtils.out_log("Rank {} not in valid rank_list {}!".format(self._rank_id, self._rank_set),
                                     DynamicProfilerUtils.LoggerLevelEnum.WARNING)
        return False

    def meta_data(self):
        if not isinstance(self._meta_data, dict):
            return {}
        return self._meta_data

    def activities(self) -> list:
        return list(self.activity_set)

    def prof_path(self) -> str:
        return self.prof_path

    def analyse(self) -> bool:
        if isinstance(self._analyse, bool):
            return self._analyse
        return False

    def async_mode(self) -> bool:
        if isinstance(self._async_mode, bool):
            return self._async_mode
        return False

    def record_shapes(self) -> bool:
        return self.record_shapes

    def profile_memory(self) -> bool:
        return self.profile_memory

    def with_stack(self) -> bool:
        return self.with_stack

    def with_flops(self) -> bool:
        return self.with_flops

    def with_modules(self) -> bool:
        return self.with_modules

    def active(self) -> int:
        if not isinstance(self._active, int) or self._active <= 0:
            DynamicProfilerUtils.out_log("Invalid parameter active, reset it to 1.",
                                         DynamicProfilerUtils.LoggerLevelEnum.WARNING)
            return self.DEFAULT_ACTIVE_NUM
        return self._active

    def warmup(self) -> int:
        if not isinstance(self._warmup, int) or self._warmup < 0:
            DynamicProfilerUtils.out_log("Invalid parameter warmup, reset it to 0.",
                                         DynamicProfilerUtils.LoggerLevelEnum.WARNING)
            return self.DEFAULT_WARMUP
        return self._warmup

    def start_step(self) -> int:
        return self._start_step

    def start(self) -> bool:
        return self._start_step == self.INSTANT_START_STEP

    def experimental_config(self) -> _ExperimentalConfig:
        return self.experimental_config

    def is_dyno_monitor(self) -> bool:
        return self._is_dyno_monitor

    @staticmethod
    def profiler_cfg_json_to_bytes(json_dict: dict) -> bytes:
        cfg_json_str = json.dumps(json_dict)
        cfg_json_bytes = cfg_json_str.encode("utf-8")
        return cfg_json_bytes

    @staticmethod
    def bytes_to_profiler_cfg_json(bytes_shm: bytes) -> dict:
        cfg_json_str = bytes_shm.decode("utf-8")
        cfg_json = json.loads(cfg_json_str)
        return cfg_json
