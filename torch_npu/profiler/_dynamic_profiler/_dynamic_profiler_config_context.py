import os
import json
import torch
from torch_npu._C._profiler import ProfilerActivity
from ..experimental_config import _ExperimentalConfig, ProfilerLevel, AiCMetrics
from ._dynamic_profiler_log import logger


class ConfigContext:
    DEFAULT_ACTIVE_NUM = 1

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
        self.rank_set = set()
        self.experimental_config = None
        self._active = 1
        self.is_valid = False
        self._meta_data = {}
        self._rank_id = self.get_rank_id()
        self.parse(json_data)

    def parse(self, json_data: dict):
        activities = json_data.get('activities')
        self.is_valid = json_data.get("is_valid", False)
        if activities and isinstance(activities, list):
            for entry in activities:
                activity = getattr(ProfilerActivity, entry.upper(), None)
                if activity:
                    self.activity_set.add(activity)
        self.prof_path = json_data.get('prof_dir')
        self._meta_data = json_data.get('metadata', {})
        self._analyse = json_data.get('analyse', False)
        self.record_shapes = json_data.get('record_shapes', False)
        self.profile_memory = json_data.get('profile_memory', False)
        self.with_stack = json_data.get('with_stack', False)
        self.with_flops = json_data.get('with_flops', False)
        self.with_modules = json_data.get('with_modules', False)
        self._active = json_data.get('active', self.DEFAULT_ACTIVE_NUM)
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
            self.parse_ranks(json_data)

    def parse_ranks(self, json_data: dict):
        self.is_rank = json_data.get("is_rank", False)
        if not isinstance(self.is_rank, bool):
            self.is_rank = False
            logger.warning("Set is_rank failed, is_rank must be bool!")
            return
        if not self.is_rank:
            return
        logger.info("Set is_rank success!")
        ranks = json_data.get("rank_list", False)
        if not isinstance(ranks, list):
            logger.warning("Set rank_list failed, rank_list must be list!")
            return
        for rank in ranks:
            if isinstance(rank, int):
                self.rank_set.add(rank)

    def valid(self) -> bool:
        if not self.is_valid:
            return False
        if not self.is_rank:
            return True
        if self._rank_id in self.rank_set:
            self._analyse = False
            logger.info("Rank {} is in valid rank_list {}, profiler data analyse will be closed !".format(
                self._rank_id, self.rank_set))
            return True
        logger.warning("Rank {} not in valid rank_list {}!".format(self._rank_id, self.rank_set))
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
            logger.warning("Invalid parameter active, reset it to 1.")
            return self.DEFAULT_ACTIVE_NUM
        return self._active

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

    @staticmethod
    def get_rank_id() -> int:
        try:
            rank_id = os.environ.get('RANK')
            if rank_id is None and torch.distributed.is_available() and torch.distributed.is_initialized():
                rank_id = torch.distributed.get_rank()
            if not isinstance(rank_id, int):
                rank_id = int(rank_id)
        except Exception as ex:
            logger.warning("Get rank id  %s, rank_id will be set to 0 !", str(ex))
            rank_id = 0

        return rank_id
