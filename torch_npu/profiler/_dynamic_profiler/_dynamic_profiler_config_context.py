import json
from torch_npu._C._profiler import ProfilerActivity
from ..experimental_config import _ExperimentalConfig, ProfilerLevel, AiCMetrics
from ._dynamic_profiler_utils import DynamicProfilerUtils


class ConfigContext:
    DEFAULT_ACTIVE_NUM = 1
    DEFAULT_START_STEP = 0
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
        self.is_valid = False
        self._meta_data = {}
        self._async_mode = False
        self._is_dyno = DynamicProfilerUtils.is_dyno_model()
        self._rank_id = DynamicProfilerUtils.get_rank_id()
        self.parse(json_data)

    def parse(self, json_data: dict):
        activities = json_data.get('activities')
        self.is_valid = json_data.get("is_valid", False)
        if activities and isinstance(activities, list):
            for entry in activities:
                activity = getattr(ProfilerActivity, entry.upper(), None)
                if activity:
                    self.activity_set.add(activity)
        self._parse_prof_dir(json_data)
        self._meta_data = json_data.get('metadata', {})
        self._analyse = json_data.get('analyse', False)
        self._async_mode = json_data.get('async_mode', False)
        self._parse_report_shape(json_data)
        self._parse_profiler_memory(json_data)
        self._parse_with_flops(json_data)
        self._parse_with_stack(json_data)
        self._parse_with_modules(json_data)
        self._parse_active(json_data)
        self._parse_start_step(json_data)
        exp_config = json_data.get('experimental_config')
        if not exp_config:
            self.experimental_config = None
        else:
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
            self.experimental_config = _ExperimentalConfig(
                profiler_level=profiler_level,
                aic_metrics=aic_metrics,
                l2_cache=l2_cache,
                op_attr=op_attr,
                gc_detect_threshold=gc_detect_threshold,
                data_simplification=data_simplification,
                record_op_args=record_op_args,
                export_type=export_type,
                msprof_tx=msprof_tx
            )
            self._parse_ranks(json_data)

    def _parse_start_step(self, json_data: dict):
        if not self._is_dyno:
            self._start_step = json_data.get("start_step", self.DEFAULT_START_STEP)
        else:
            start_step = json_data.get("PROFILE_START_ITERATION_ROUNDUP", self.DEFAULT_START_STEP)
            try:
                self._start_step = int(start_step)
            except ValueError:
                self._start_step = self.DEFAULT_START_STEP

        if not isinstance(self._start_step, int) or self._start_step < 0:
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
            record_shapes = json_data.get("PROFILE_REPORT_INPUT_SHAPES")
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

    def valid(self) -> bool:
        if not self.is_valid:
            return False
        if not self.is_rank:
            return True
        if self._rank_id in self._rank_set:
            self._analyse = False
            DynamicProfilerUtils.out_log("Rank {} is in rank_list {}, profiler data analyse will be closed !".format(
                self._rank_id, self._rank_set), DynamicProfilerUtils.LoggerLevelEnum.INFO)
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

    def start_step(self) -> int:
        return self._start_step

    def experimental_config(self) -> _ExperimentalConfig:
        return self.experimental_config

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
