import os
import mmap
import stat
import time
import json
import struct
import multiprocessing

from ._dynamic_profiler_log import logger, logger_monitor, init_logger
from ._dynamic_profiler_config_context import ConfigContext
from ._dynamic_profiler_monitor_shm import DynamicProfilerShareMemory


class DynamicProfilerMonitor:
    def __init__(
            self,
            path: str,
            buffer_size: int = 1024,
            poll_interval: int = 2
    ):
        self._path = path
        self._rank_id = ConfigContext.get_rank_id()
        self._buffer_size = buffer_size
        self._monitor_process = None
        self.prof_cfg_context = None
        self._shared_loop_flag = multiprocessing.Value('b', True)
        self._step_time = multiprocessing.Value('i', poll_interval)
        self._config_path = os.path.join(self._path, 'profiler_config.json')
        self._shm_obj = DynamicProfilerShareMemory(
            self._path,
            self._config_path,
            self._rank_id,
            self._buffer_size)
        self._cur_time = int(time.time())
        self._create_process()

    def shm_to_prof_conf_context(self):
        if self._shm_obj is None:
            logger.warning('Rank %d shared memory is None !', self._rank_id)
            return None
        try:
            time_bytes_data = self._shm_obj.read_bytes(read_time=True)
            shm_cfg_change_time = struct.unpack("<I", time_bytes_data)[0]
        except Exception as ex:
            logger.error("Share memory read error: %s", ex)
            return None

        if shm_cfg_change_time <= self._cur_time:
            return None
        self._cur_time = shm_cfg_change_time
        try:
            json_data = ConfigContext.bytes_to_profiler_cfg_json(self._shm_obj.read_bytes())
        except Exception as ex:
            logger.error("Share memory bytes to json error: %s", ex)
            return None

        self.prof_cfg_context = ConfigContext(json_data)
        if not self.prof_cfg_context.valid():
            return None
        return self.prof_cfg_context

    def clean_resource(self):
        if self._process is not None:
            self._shared_loop_flag.value = False
            self._process.join()
            self._shm_obj.clean_resource()

    def modify_step_time(self, poll_interval_time: int):
        self._step_time.value = poll_interval_time
        logger.info("Dynamic profiling monitor process query cfg file interval time change to %ds", poll_interval_time)

    def _monitor_process_params(self):
        shm = None if self._shm_obj.is_mmap else self._shm_obj
        mmap_path = self._shm_obj.shm_path if self._shm_obj.is_mmap else None
        params = {
            "loop_flag": self._shared_loop_flag,
            "poll_interval": self._step_time,
            "shm": shm,
            "cfg_path": self._shm_obj.config_path,
            "max_size": self._buffer_size,
            "file_stat_time": self._shm_obj.cur_mtime,
            "mmap_path": mmap_path,
            "is_mmap": self._shm_obj.is_mmap
        }
        return params

    def _create_process(self):
        """Create json monitor process, one process will be created at one worker"""
        if self._shm_obj.is_create_process:
            process_params = self._monitor_process_params()
            # daemon need to be set to True, otherwise the process will not be killed when the main process exits.
            self._process = multiprocessing.Process(
                target=worker_func, daemon=True,
                args=(process_params, ))
            self._process.start()
            logger.info("Config monitor process has been created by rank %d.",
                        self._rank_id)
        else:
            self._process = None
            logger.info("Rank %d no need to create process.", self._rank_id)


def worker_func(params_dict):
    """ Json monitor process worker function python version >= 3.8"""
    loop_flag = params_dict.get("loop_flag")
    poll_interval = params_dict.get("poll_interval")
    shm = params_dict.get("shm")
    cfg_path = params_dict.get("cfg_path")
    max_size = params_dict.get("max_size")
    file_stat_time = params_dict.get("file_stat_time")
    mmap_path = params_dict.get("mmap_path")
    is_mmap = params_dict.get("is_mmap")
    init_logger(logger_monitor, os.path.dirname(cfg_path), True)

    mmap_obj = None
    if is_mmap and mmap_path is not None:
        try:
            fd = os.open(mmap_path, os.O_EXCL | os.O_RDWR, stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP)
            f = os.fdopen(fd, 'rb')
            mmap_obj = mmap.mmap(f.fileno(), length=max_size)
        except Exception as ex:
            logger_monitor.warning("Dynamic profiler process start failed, %s occurred!", str(ex))
            return

    last_file_t = file_stat_time
    while loop_flag.value:
        if os.path.exists(cfg_path):
            file_t = int(os.path.getmtime(cfg_path))
            if not last_file_t or last_file_t != file_t:
                last_file_t = file_t

                try:
                    with open(cfg_path, 'r') as f:
                        data = json.load(f)
                    # convert json to bytes
                    data['is_valid'] = True
                    logger_monitor.info("Dynamic profiler process load json success")

                except Exception as ex:
                    data = {'is_valid': False}
                    logger_monitor.warning("Dynamic profiler process load json failed, %s has occur!", str(ex))
                time_bytes = struct.pack("<I", last_file_t)
                prof_cfg_bytes = time_bytes + ConfigContext.profiler_cfg_json_to_bytes(data)
                if len(prof_cfg_bytes) > max_size:
                    logger_monitor.warning("Dynamic profiler process load json failed, "
                                           "because cfg bytes size over %d bytes", max_size)
                    continue
                try:
                    if is_mmap and mmap is not None:
                        DynamicProfilerShareMemory.write_bytes_py37(mmap_obj, prof_cfg_bytes, max_size)
                    elif shm is not None:
                        shm.write_bytes_over_py38(prof_cfg_bytes)
                except Exception as ex:
                    logger_monitor.warning("Dynamic profiler cfg bytes write failed, %s has occur!", str(ex))
        else:
            logger_monitor.error("Dynamic profiler cfg json not exists")
        time.sleep(poll_interval.value)
    logger_monitor.info("Dynamic profiler process done")
