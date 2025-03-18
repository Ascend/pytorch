import os
import mmap
import stat
import time
import struct
import multiprocessing
from ...utils._path_manager import PathManager
from ..analysis.prof_common_func._file_manager import FileManager
from ._dynamic_profiler_config_context import ConfigContext
from ._dynamic_profiler_utils import DynamicProfilerUtils
from ._dynamic_profiler_monitor_shm import DynamicProfilerShareMemory
from ._dynamic_monitor_proxy import PyDynamicMonitorProxySingleton


class DynamicProfilerMonitor:
    def __init__(
            self
    ):
        self._path = DynamicProfilerUtils.CFG_CONFIG_PATH
        self._rank_id = DynamicProfilerUtils.get_rank_id()
        self._buffer_size = DynamicProfilerUtils.CFG_BUFFER_SIZE
        self._monitor_process = None
        self.prof_cfg_context = None
        self._shared_loop_flag = multiprocessing.Value('b', True)
        self._step_time = multiprocessing.Value('i', DynamicProfilerUtils.POLL_INTERVAL)
        self._config_path = None
        self._is_dyno = DynamicProfilerUtils.is_dyno_model()
        if not self._is_dyno:
            self._config_path = os.path.join(self._path, 'profiler_config.json')
        self._shm_obj = DynamicProfilerShareMemory(
            self._path,
            self._config_path,
            self._rank_id)
        self._cur_time = int(time.time())
        self._create_process()

    def shm_to_prof_conf_context(self):
        if self._shm_obj is None:
            DynamicProfilerUtils.out_log('Rank {} shared memory is None !'.format(
                self._rank_id), DynamicProfilerUtils.LoggerLevelEnum.ERROR)
            return None
        try:
            time_bytes_data = self._shm_obj.read_bytes(read_time=True)
            shm_cfg_change_time = struct.unpack("<I", time_bytes_data)[0]
        except Exception as ex:
            DynamicProfilerUtils.out_log("Share memory read error: {}".format(
                str(ex)), DynamicProfilerUtils.LoggerLevelEnum.WARNING)
            return None

        if shm_cfg_change_time <= self._cur_time:
            return None
        self._cur_time = shm_cfg_change_time
        try:
            json_data = ConfigContext.bytes_to_profiler_cfg_json(self._shm_obj.read_bytes())
        except Exception as ex:
            DynamicProfilerUtils.out_log("Share memory bytes to json error: {}".format(
                str(ex)), DynamicProfilerUtils.LoggerLevelEnum.ERROR)
            return None

        self.prof_cfg_context = ConfigContext(json_data)
        if not self.prof_cfg_context.valid():
            return None
        if self.prof_cfg_context.is_dyno_monitor():
            self._call_dyno_monitor(json_data)
            return None
        return self.prof_cfg_context

    def clean_resource(self):
        if self._process is not None:
            self._shared_loop_flag.value = False
            self._process.join()
            self._shm_obj.clean_resource()

    def modify_step_time(self, poll_interval_time: int):
        self._step_time.value = poll_interval_time
        DynamicProfilerUtils.out_log("Dynamic profiling monitor process poll interval time change to {}s".format(
            poll_interval_time), DynamicProfilerUtils.LoggerLevelEnum.INFO)

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
            "is_mmap": self._shm_obj.is_mmap,
            "rank_id": self._rank_id,
            "dynamic_profiler_utils": DynamicProfilerUtils
        }
        return params

    def _create_process(self):
        """Create json monitor process, one process will be created at one worker"""
        if self._shm_obj.is_create_process:
            process_params = self._monitor_process_params()
            # daemon need to be set to True, otherwise the process will not be killed when the main process exits.
            self._process = multiprocessing.Process(
                target=worker_func if not self._is_dyno else worker_dyno_func, daemon=True,
                args=(process_params, ))
            self._process.start()
            DynamicProfilerUtils.out_log("Dynamic monitor process has been created by rank {}.".format(
                self._rank_id), DynamicProfilerUtils.LoggerLevelEnum.INFO)
        else:
            self._process = None
            DynamicProfilerUtils.out_log("Rank {} no need to create process.".format(
                self._rank_id), DynamicProfilerUtils.LoggerLevelEnum.INFO)

    def _call_dyno_monitor(self, json_data: dict):
        json_data = {key: str(value) for key, value in json_data.items()}
        py_dyno_monitor = PyDynamicMonitorProxySingleton().get_proxy()
        if py_dyno_monitor:
            py_dyno_monitor.enable_dyno_npu_monitor(json_data)


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
    dynamic_profiler_utils = params_dict.get("dynamic_profiler_utils")
    dynamic_profiler_utils.init_logger(is_monitor_process=True)
    mmap_obj = None
    if is_mmap and mmap_path is not None:
        try:
            fd = os.open(mmap_path, os.O_EXCL | os.O_RDWR, stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP)
            f = os.fdopen(fd, 'rb')
            mmap_obj = mmap.mmap(f.fileno(), length=max_size)
        except Exception as ex:
            dynamic_profiler_utils.out_log("Dynamic profiler process start failed, {} occurred!".format(str(ex)),
                                           dynamic_profiler_utils.LoggerLevelEnum.ERROR, is_monitor_process=True)
            return
    last_file_t = file_stat_time
    while loop_flag.value:
        if os.path.exists(cfg_path):
            file_t = int(os.path.getmtime(cfg_path))
            if not last_file_t or last_file_t != file_t:
                last_file_t = file_t
                try:
                    PathManager.check_input_file_path(cfg_path)
                    PathManager.check_directory_path_readable(cfg_path)
                    data = FileManager.read_json_file(cfg_path)
                    # convert json to bytes
                    data['is_valid'] = True
                    DynamicProfilerUtils.out_log("Dynamic profiler process load json success",
                                                 DynamicProfilerUtils.LoggerLevelEnum.INFO, is_monitor_process=True)
                except Exception as ex:
                    data = {'is_valid': False}
                    dynamic_profiler_utils.out_log("Dynamic profiler process load json failed, {} has occur!".format(
                        str(ex)), dynamic_profiler_utils.LoggerLevelEnum.ERROR, is_monitor_process=True)
                time_bytes = struct.pack("<I", last_file_t)
                prof_cfg_bytes = time_bytes + ConfigContext.profiler_cfg_json_to_bytes(data)
                if len(prof_cfg_bytes) > max_size:
                    dynamic_profiler_utils.out_log("Load json failed,  because cfg bytes size over {} bytes".format(
                        max_size), DynamicProfilerUtils.LoggerLevelEnum.WARNING, is_monitor_process=True)
                    continue
                try:
                    if is_mmap and mmap is not None:
                        DynamicProfilerShareMemory.write_bytes_py37(mmap_obj, prof_cfg_bytes, max_size)
                    elif shm is not None:
                        shm.write_bytes_over_py38(prof_cfg_bytes)
                except Exception as ex:
                    dynamic_profiler_utils.out_log("Dynamic profiler cfg bytes write failed, {} has occur!".format(
                        str(ex)), dynamic_profiler_utils.LoggerLevelEnum.ERROR, is_monitor_process=True)
        else:
            dynamic_profiler_utils.out_log("Dynamic profiler cfg json not exists",
                                           dynamic_profiler_utils.LoggerLevelEnum.ERROR, is_monitor_process=True)
        time.sleep(poll_interval.value)
    dynamic_profiler_utils.out_log("Dynamic profiler process done", dynamic_profiler_utils.LoggerLevelEnum.INFO,
                                   is_monitor_process=True)


def worker_dyno_func(params_dict):
    """ Json monitor process worker function python version >= 3.8"""
    loop_flag = params_dict.get("loop_flag")
    poll_interval = params_dict.get("poll_interval")
    shm = params_dict.get("shm")
    rank_id = params_dict.get("rank_id")
    max_size = params_dict.get("max_size")
    dynamic_profiler_utils = params_dict.get("dynamic_profiler_utils")

    py_dyno_monitor = PyDynamicMonitorProxySingleton().get_proxy()
    if not py_dyno_monitor:
        return
    ret = py_dyno_monitor.init_dyno(rank_id)
    if not ret:
        dynamic_profiler_utils.out_log("Init dynolog failed !", dynamic_profiler_utils.LoggerLevelEnum.WARNING)
        return
    dynamic_profiler_utils.out_log("Init dynolog success !", dynamic_profiler_utils.LoggerLevelEnum.INFO)
    while loop_flag.value:
        time.sleep(poll_interval.value)
        res = py_dyno_monitor.poll_dyno()
        data = DynamicProfilerUtils.dyno_str_to_json(res)
        if data:
            data['is_valid'] = True
            dynamic_profiler_utils.out_log("Dynolog profiler process load json success",
                                         dynamic_profiler_utils.LoggerLevelEnum.INFO)
        else:
            continue
        time_bytes = struct.pack("<I", int(time.time()))
        prof_cfg_bytes = time_bytes + ConfigContext.profiler_cfg_json_to_bytes(data)
        if len(prof_cfg_bytes) > max_size:
            dynamic_profiler_utils.out_log("Load json failed, because cfg bytes size over {} bytes".format(
                max_size), dynamic_profiler_utils.LoggerLevelEnum.INFO)
            continue
        try:
            if shm is not None:
                shm.write_bytes_over_py38(prof_cfg_bytes)
        except Exception as ex:
            dynamic_profiler_utils.out_log("Dynamic profiler cfg bytes write failed, {} has occur!".format(str(ex)),
                                           dynamic_profiler_utils.LoggerLevelEnum.ERROR)
    dynamic_profiler_utils.out_log("Dynolog profiler process done", dynamic_profiler_utils.LoggerLevelEnum.INFO)
