import mmap
import multiprocessing
import os
import stat
import struct
import time
import uuid

import torch.multiprocessing as mp

from ...utils._path_manager import PathManager
from ..analysis.prof_common_func._file_manager import FileManager
from ._dynamic_monitor_proxy import PyDynamicMonitorProxySingleton
from ._dynamic_profiler_config_context import ConfigContext
from ._dynamic_profiler_monitor_shm import DynamicProfilerShareMemory
from ._dynamic_profiler_utils import DynamicProfilerUtils


MAX_RETRY_CONFIGS = 64
MAX_PENDING_CONFIGS = 64


def _shm_payload_fits(cfg_list: list, time_bytes: bytes, max_size: int) -> bool:
    return len(time_bytes + ConfigContext.profiler_cfg_json_to_bytes(cfg_list)) <= max_size


def _max_fitting_prefix(cfg_list: list, time_bytes: bytes, max_size: int) -> int:
    low, high = 1, len(cfg_list)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        if _shm_payload_fits(cfg_list[:mid], time_bytes, max_size):
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


class DynamicProfilerMonitor:
    def __init__(self):
        self._path = DynamicProfilerUtils.CFG_CONFIG_PATH
        self._rank_id = DynamicProfilerUtils.get_rank_id()
        self._buffer_size = DynamicProfilerUtils.CFG_BUFFER_SIZE
        self._monitor_process = None
        self.prof_cfg_context = None
        self._shared_loop_flag = multiprocessing.Value("b", True)
        self._step_time = multiprocessing.Value("i", DynamicProfilerUtils.POLL_INTERVAL)
        self._profiler_status = {
            DynamicProfilerUtils.PROFILER_STATUS: str(
                DynamicProfilerUtils.ProfilerStatus.IDLE.value
            ),
            DynamicProfilerUtils.CURRENT_STEP: "-1",
            DynamicProfilerUtils.START_STEP: "-1",
            DynamicProfilerUtils.STOP_STEP: "-1",
            DynamicProfilerUtils.TRACE_ID: "",
            DynamicProfilerUtils.QUEUE_DEPTH: "0",
            DynamicProfilerUtils.REJECTED: "0",
        }
        self._last_report_time = 0.0
        self._config_path = None
        self._is_dyno = DynamicProfilerUtils.is_dyno_model()
        if not self._is_dyno:
            self._config_path = os.path.join(self._path, "profiler_config.json")
        self._shm_obj = DynamicProfilerShareMemory(
            self._path, self._config_path, self._rank_id
        )
        self._cur_time = int(time.time())
        self._consume_seq = multiprocessing.Value("i", 0)
        self._create_process()

    def shm_to_prof_conf_context(self):
        if self._shm_obj is None:
            DynamicProfilerUtils.out_log(
                f"Rank {self._rank_id} shared memory is None !",
                DynamicProfilerUtils.LoggerLevelEnum.ERROR)
            return []
        try:
            time_bytes_data = self._shm_obj.read_bytes(read_time=True)
            shm_cfg_change_time = struct.unpack("<I", time_bytes_data)[0]
        except Exception as ex:
            DynamicProfilerUtils.out_log(
                f"Share memory read error: {str(ex)}",
                DynamicProfilerUtils.LoggerLevelEnum.WARNING)
            return []
        if shm_cfg_change_time <= self._cur_time:
            return []
        try:
            json_data = ConfigContext.bytes_to_profiler_cfg_json(self._shm_obj.read_bytes())
        except Exception as ex:
            DynamicProfilerUtils.out_log(
                f"Share memory bytes to json error: {str(ex)}",
                DynamicProfilerUtils.LoggerLevelEnum.ERROR)
            return []
        self._cur_time = shm_cfg_change_time
        self._consume_seq.value += 1
        cfg_ctx_list = []
        cfg_dict_list = json_data if isinstance(json_data, list) else [json_data]
        for cfg_dict in cfg_dict_list:
            if not isinstance(cfg_dict, dict):
                DynamicProfilerUtils.out_log(
                    f"Share memory json data is not valid, expect dict, but got {type(cfg_dict)}",
                    DynamicProfilerUtils.LoggerLevelEnum.WARNING)
                continue
            try:
                cfg_ctx = ConfigContext(cfg_dict)
                if cfg_ctx.valid():
                    if cfg_ctx.is_dyno_monitor():
                        self._call_dyno_monitor(cfg_dict)
                        continue
                    cfg_ctx_list.append(cfg_ctx)
            except Exception as ex:
                DynamicProfilerUtils.out_log(
                    f"Skip invalid config context, trace id = "
                    f"{cfg_dict.get('REQUEST_TRACE_ID', 'N/A')}, error: {str(ex)}",
                    DynamicProfilerUtils.LoggerLevelEnum.ERROR)
        self.prof_cfg_context = cfg_ctx_list
        return cfg_ctx_list

    def clean_resource(self):
        if self._process is not None:
            self._shared_loop_flag.value = False
            self._process.join()
            self._shm_obj.clean_resource()

    def modify_step_time(self, poll_interval_time: int):
        self._step_time.value = poll_interval_time
        DynamicProfilerUtils.out_log(
            f"Dynamic profiling monitor process poll interval time change to {poll_interval_time}s",
            DynamicProfilerUtils.LoggerLevelEnum.INFO,
        )

    def update_profiler_status(self, status: DynamicProfilerUtils.ProfilerStatus):
        self._profiler_status[DynamicProfilerUtils.PROFILER_STATUS] = str(
            status.value
        )  # 业务保证不会触发 KeyError

    def update_cfg_status(self, trace_id: str, queue_depth: int, rejected: int):
        self._profiler_status[DynamicProfilerUtils.TRACE_ID] = trace_id or ""
        self._profiler_status[DynamicProfilerUtils.QUEUE_DEPTH] = str(queue_depth)
        self._profiler_status[DynamicProfilerUtils.REJECTED] = str(rejected)

    def update_step_info(self, step: int, flag: str):
        if flag in self._profiler_status:
            self._profiler_status[flag] = str(step)

    def report_profiler_status(self):
        if not self._shm_obj.is_create_process:
            return
        if not self._is_dyno:
            return
        current_time = time.time()
        if current_time - self._last_report_time < DynamicProfilerUtils.REPORT_INTERVAL:
            return
        py_dyno_monitor = PyDynamicMonitorProxySingleton().get_proxy()
        if not py_dyno_monitor or not hasattr(
            py_dyno_monitor, "update_profiler_status"
        ):
            return
        py_dyno_monitor.update_profiler_status(self._profiler_status)
        self._last_report_time = current_time

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
            "consume_seq": self._consume_seq,
            "dynamic_profiler_utils": DynamicProfilerUtils,
        }
        return params

    def _create_process(self):
        """Create json monitor process, one process will be created at one worker"""
        if self._shm_obj.is_create_process:
            process_params = self._monitor_process_params()
            # daemon need to be set to True, otherwise the process will not be killed when the main process exits.
            ctx = multiprocessing.get_context("fork")
            self._process = ctx.Process(
                target=worker_func if not self._is_dyno else worker_dyno_func,
                daemon=True,
                args=(process_params,),
            )
            self._process.start()
            DynamicProfilerUtils.out_log(
                f"Dynamic monitor process has been created by rank {self._rank_id}.",
                DynamicProfilerUtils.LoggerLevelEnum.INFO,
            )
        else:
            self._process = None
            DynamicProfilerUtils.out_log(
                f"Rank {self._rank_id} no need to create process.",
                DynamicProfilerUtils.LoggerLevelEnum.INFO,
            )

    def _call_dyno_monitor(self, json_data: dict):
        json_data = {key: str(value) for key, value in json_data.items()}
        py_dyno_monitor = PyDynamicMonitorProxySingleton().get_proxy()
        if py_dyno_monitor:
            py_dyno_monitor.enable_dyno_npu_monitor(json_data)


def worker_func(params_dict):
    """Json monitor process worker function python version >= 3.8"""
    mp._set_thread_name("DynMonitorProc")
    loop_flag = params_dict.get("loop_flag")
    poll_interval = params_dict.get("poll_interval")
    shm = params_dict.get("shm")
    cfg_path = params_dict.get("cfg_path")
    max_size = params_dict.get("max_size")
    file_stat_time = params_dict.get("file_stat_time")
    mmap_path = params_dict.get("mmap_path")
    is_mmap = params_dict.get("is_mmap")
    dynamic_profiler_utils = params_dict.get("dynamic_profiler_utils")
    dynamic_profiler_utils.init_logger(
        is_monitor_process=True, log_path=os.path.dirname(cfg_path)
    )
    mmap_obj = None
    if is_mmap and mmap_path is not None:
        try:
            fd = os.open(
                mmap_path,
                os.O_EXCL | os.O_RDWR,
                stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP,
            )
            f = os.fdopen(fd, "rb")
            mmap_obj = mmap.mmap(f.fileno(), length=max_size)
        except Exception as ex:
            dynamic_profiler_utils.out_log(
                f"Dynamic profiler process start failed, {str(ex)} occurred!",
                dynamic_profiler_utils.LoggerLevelEnum.ERROR,
                is_monitor_process=True,
            )
            return
    last_file_t = file_stat_time
    while loop_flag.value:
        if os.path.exists(cfg_path):
            try:
                file_t = int(os.path.getmtime(cfg_path))
            except Exception as ex:
                dynamic_profiler_utils.out_log(
                    f"Dynamic profiler process get json modify time failed, {str(ex)} has occur!",
                    dynamic_profiler_utils.LoggerLevelEnum.ERROR,
                    is_monitor_process=True,
                )
                continue
            if not last_file_t or last_file_t != file_t:
                last_file_t = file_t
                try:
                    PathManager.check_input_file_path(cfg_path)
                    PathManager.check_directory_path_readable(cfg_path)
                    data = FileManager.read_json_file(cfg_path)
                    # convert json to bytes
                    data["is_valid"] = True
                    dynamic_profiler_utils.out_log(
                        "Dynamic profiler process load json success",
                        dynamic_profiler_utils.LoggerLevelEnum.INFO,
                        is_monitor_process=True,
                    )
                except Exception as ex:
                    data = {"is_valid": False}
                    dynamic_profiler_utils.out_log(
                        f"Dynamic profiler process load json failed, {str(ex)} has occur!",
                        dynamic_profiler_utils.LoggerLevelEnum.ERROR,
                        is_monitor_process=True,
                    )
                time_bytes = struct.pack("<I", last_file_t)
                prof_cfg_bytes = time_bytes + ConfigContext.profiler_cfg_json_to_bytes(
                    data
                )
                if len(prof_cfg_bytes) > max_size:
                    dynamic_profiler_utils.out_log(
                        f"Load json failed,  because cfg bytes size over {max_size} bytes",
                        DynamicProfilerUtils.LoggerLevelEnum.WARNING,
                        is_monitor_process=True,
                    )
                    continue
                try:
                    if is_mmap and mmap_obj is not None:
                        DynamicProfilerShareMemory.write_bytes_py37(
                            mmap_obj, prof_cfg_bytes, max_size
                        )
                    elif shm is not None:
                        shm.write_bytes_over_py38(prof_cfg_bytes)
                except Exception as ex:
                    dynamic_profiler_utils.out_log(
                        f"Dynamic profiler cfg bytes write failed, {str(ex)} has occur!",
                        dynamic_profiler_utils.LoggerLevelEnum.ERROR,
                        is_monitor_process=True,
                    )
        else:
            dynamic_profiler_utils.out_log(
                "Dynamic profiler cfg json not exists",
                dynamic_profiler_utils.LoggerLevelEnum.ERROR,
                is_monitor_process=True,
            )
        time.sleep(poll_interval.value)
    dynamic_profiler_utils.out_log(
        "Dynamic profiler process done",
        dynamic_profiler_utils.LoggerLevelEnum.INFO,
        is_monitor_process=True,
    )


def worker_dyno_func(params_dict):
    """Json monitor process worker function python version >= 3.8"""
    mp._set_thread_name("DynMonitorProc")
    loop_flag = params_dict.get("loop_flag")
    poll_interval = params_dict.get("poll_interval")
    shm = params_dict.get("shm")
    rank_id = params_dict.get("rank_id")
    max_size = params_dict.get("max_size")
    consume_seq = params_dict.get("consume_seq")
    dynamic_profiler_utils = params_dict.get("dynamic_profiler_utils")

    py_dyno_monitor = PyDynamicMonitorProxySingleton().get_proxy()
    if not py_dyno_monitor:
        return
    if not py_dyno_monitor.init_dyno(rank_id):
        dynamic_profiler_utils.out_log(
            "Init dynolog failed !", dynamic_profiler_utils.LoggerLevelEnum.WARNING)
        return
    dynamic_profiler_utils.out_log(
        "Init dynolog success !", dynamic_profiler_utils.LoggerLevelEnum.INFO)

    pending_configs = []
    retry_configs = []
    last_write_time = 0
    consume_at_write = consume_seq.value
    while loop_flag.value:
        time.sleep(poll_interval.value)
        if pending_configs and consume_seq.value > consume_at_write:
            dynamic_profiler_utils.out_log(
                f"Main process consumed dynolog config batch, clear {len(pending_configs)} cached config(s).",
                dynamic_profiler_utils.LoggerLevelEnum.INFO)
            pending_configs.clear()

        res = py_dyno_monitor.poll_dyno()
        try:
            if isinstance(res, (bytes, bytearray)):
                res = bytes(res).decode("utf-8", errors="replace")
            new_configs = DynamicProfilerUtils.parse_dyno_configs(res) if res else []
            for cfg in new_configs:
                cfg["is_valid"] = True
                if not cfg.get("REQUEST_TRACE_ID"):
                    cfg["REQUEST_TRACE_ID"] = "dyno-{}".format(uuid.uuid4().hex)
        except Exception as ex:
            dynamic_profiler_utils.out_log(
                f"Parse dynolog config failed, {str(ex)} has occur!",
                dynamic_profiler_utils.LoggerLevelEnum.ERROR)
            new_configs = []

        to_send = retry_configs + new_configs
        retry_configs = []
        if not to_send:
            continue

        deliverable = []
        for cfg in to_send:
            if len(ConfigContext.profiler_cfg_json_to_bytes([cfg])) + 4 > max_size:
                dynamic_profiler_utils.out_log(
                    f"Drop dynolog config that exceeds shm size ({max_size} bytes), "
                    f"trace id = {cfg.get('REQUEST_TRACE_ID', 'N/A')}",
                    dynamic_profiler_utils.LoggerLevelEnum.ERROR)
                continue
            deliverable.append(cfg)
        if not deliverable:
            continue
        if len(deliverable) < len(to_send):
            dynamic_profiler_utils.out_log(
                f"Dropped {len(to_send) - len(deliverable)} oversized config(s), keep {len(deliverable)} config(s) to send.",
                dynamic_profiler_utils.LoggerLevelEnum.ERROR)
        to_send = deliverable

        if shm is None:
            continue

        write_time = int(time.time())
        if write_time <= last_write_time:
            write_time = last_write_time + 1
        time_bytes = struct.pack("<I", write_time)

        pending_count = len(pending_configs)
        if pending_count and not _shm_payload_fits(
                pending_configs, time_bytes, max_size):
            dynamic_profiler_utils.out_log(
                f"Cached dynolog config batch exceeds shm size ({max_size} bytes), drop {pending_count} cached config(s).",
                dynamic_profiler_utils.LoggerLevelEnum.ERROR)
            pending_configs = []
            pending_count = 0

        batch = pending_configs + to_send
        send_count = max(
            _max_fitting_prefix(batch, time_bytes, max_size), pending_count)
        if send_count < len(batch):
            retry_configs = batch[send_count:]
            dynamic_profiler_utils.out_log(
                f"Dynolog config batch over {max_size} bytes, keep {len(retry_configs)} config(s) for retry, "
                f"trace ids = {[cfg.get('REQUEST_TRACE_ID', 'N/A') for cfg in retry_configs[:10]]}",
                dynamic_profiler_utils.LoggerLevelEnum.WARNING)
            if len(retry_configs) > MAX_RETRY_CONFIGS:
                dynamic_profiler_utils.out_log(
                    f"Retry buffer full, drop {len(retry_configs) - MAX_RETRY_CONFIGS} oldest config(s).",
                    dynamic_profiler_utils.LoggerLevelEnum.ERROR)
                retry_configs = retry_configs[-MAX_RETRY_CONFIGS:]
        batch = batch[:send_count]
        try:
            shm.write_bytes_over_py38(
                time_bytes + ConfigContext.profiler_cfg_json_to_bytes(batch))
        except Exception as ex:
            dynamic_profiler_utils.out_log(
                f"Dynamic profiler cfg bytes write failed, {str(ex)} has occur!",
                dynamic_profiler_utils.LoggerLevelEnum.ERROR)
            retry_configs = to_send[-MAX_RETRY_CONFIGS:]
            continue

        pending_configs = batch
        if len(pending_configs) > MAX_PENDING_CONFIGS:
            drop_count = len(pending_configs) - MAX_PENDING_CONFIGS
            dropped_ids = [
                cfg.get("REQUEST_TRACE_ID", "N/A")
                for cfg in pending_configs[:drop_count]]
            del pending_configs[:drop_count]
            dynamic_profiler_utils.out_log(
                f"Pending dynolog config buffer is full (limit {MAX_PENDING_CONFIGS}), "
                f"drop {drop_count} oldest config(s), trace ids = {dropped_ids[:10]}",
                dynamic_profiler_utils.LoggerLevelEnum.ERROR)
        last_write_time = write_time
        consume_at_write = consume_seq.value
    if hasattr(py_dyno_monitor, "update_profiler_status"):
        profiler_status = {
            DynamicProfilerUtils.PROFILER_STATUS: str(
                DynamicProfilerUtils.ProfilerStatus.UNINITIALIZED.value),
            DynamicProfilerUtils.CURRENT_STEP: "-1",
            DynamicProfilerUtils.START_STEP: "-1",
            DynamicProfilerUtils.STOP_STEP: "-1",
            DynamicProfilerUtils.TRACE_ID: "",
            DynamicProfilerUtils.QUEUE_DEPTH: "0",
            DynamicProfilerUtils.REJECTED: "0",
        }
        py_dyno_monitor.update_profiler_status(profiler_status)
    dynamic_profiler_utils.out_log(
        "Dynolog profiler process done", dynamic_profiler_utils.LoggerLevelEnum.INFO)
