import os
import pickle
import random
import sys
import stat
import mmap
import time
import struct
from datetime import datetime

from ...utils.path_manager import PathManager
from ...utils._error_code import ErrCode, prof_error
from ..analysis.prof_common_func._file_manager import FileManager
from ._dynamic_profiler_log import logger


class DynamicProfilerShareMemory:
    JSON_DATA = {
        "activities": ["CPU", "NPU"],
        "prof_dir": "./",
        "analyse": False,
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
        "with_flops": False,
        "with_modules": False,
        "active": 1,
        "is_rank": False,
        "rank_list": [],
        "experimental_config": {
            "profiler_level": "Level0",
            "aic_metrics": "AiCoreNone",
            "l2_cache": False,
            "op_attr": False,
            "gc_detect_threshold": None,
            "data_simplification": True,
            "record_op_args": False,
            "export_type": "text",
            "msprof_tx": False
        }
    }

    def __init__(
            self,
            path: str,
            config_path: str,
            rank_id: int,
            buffer_size: int = 1024
    ):
        self._path = path
        self.config_path = config_path
        self._rank_id = rank_id
        self.shm_path = f"DynamicProfileNpuShm{datetime.utcnow().strftime('%Y%m%d%H')}"
        self._shm_buf_bytes_size = buffer_size
        self.is_create_process = False
        self.shm = None
        self.cur_mtime = 0
        self.is_mmap = False
        self._time_data_bytes = struct.pack("<I", self.cur_mtime)
        self._time_bytes_size = len(self._time_data_bytes)
        self._clean_shm_for_killed()
        self._create_shm()
        if self.is_create_process:
            self._create_prof_cfg()

    @staticmethod
    def _get_pid_st_ctime(pid):
        try:
            fd = os.open("/proc/" + str(pid), os.O_RDONLY, stat.S_IRUSR | stat.S_IRGRP)
            stat_ino = os.fstat(fd)
            os.close(fd)
            create_time = stat_ino.st_ctime
            return create_time
        except Exception as ex:
            logger.warning("An error is occurred: %s", str(ex))
            return None

    def _clean_shm_for_killed(self):
        if sys.version_info >= (3, 8):
            shm_path = os.path.join("/dev/shm", self.shm_path)
        else:
            shm_path = os.path.join(self._path, "shm", self.shm_path)
        if not os.path.exists(shm_path):
            return
        time_shm = os.stat(shm_path).st_ctime
        pid_time = self._get_pid_st_ctime(os.getpid())
        eps = 60
        if pid_time is not None and pid_time - time_shm > eps:
            raise RuntimeError(f"There may exist shared memory before this task. If you kill the last task, "
                               f"dynamic profiler will not be valid. Please remove: {shm_path}, and retry." +
                               prof_error(ErrCode.VALUE)) from err

    def _create_prof_cfg(self):
        if not os.path.exists(self.config_path):
            logger.info("Create profiler_config.json default.")
            FileManager.create_json_file_by_path(
                self.config_path,
                self.JSON_DATA,
                indent=4)

        file_stat = os.stat(self.config_path)
        self.cur_mtime = int(file_stat.st_mtime)

    def _create_shm(self):
        if sys.version_info >= (3, 8):
            PathManager.check_input_directory_path(self.shm_path)
            self._create_shm_over_py38()
        else:
            self.is_mmap = True
            self.shm_path = os.path.join(self._path, "shm", self.shm_path)
            PathManager.check_input_directory_path(self.shm_path)
            self._create_shm_py37()

    def _get_default_cfg_bytes(self):
        bytes_data = pickle.dumps(self.JSON_DATA)
        bytes_data = self._time_data_bytes + bytes_data
        bytes_data = bytes_data.ljust(self._shm_buf_bytes_size)
        return bytes_data

    def _create_shm_over_py38(self):
        """Create a json monitor process based on whether the SharedMemory is successfully created py38"""
        from unittest.mock import patch
        from multiprocessing import shared_memory
        try_times = 10
        while try_times:
            try:
                # Step 1: try to open shm file, first time shm not exists.
                with patch("multiprocessing.resource_tracker.register",
                           lambda *args, **kwargs: None):
                    self.shm = shared_memory.SharedMemory(name=self.shm_path)
                self.is_create_process = False
                logger.info("Rank %d shared memory is connected.", self._rank_id)
                break
            except FileNotFoundError:
                try:
                    # Step 2: only one process can create shm successfully.
                    self.shm = shared_memory.SharedMemory(name=self.shm_path, create=True,
                                                          size=self._shm_buf_bytes_size)
                    self.is_create_process = True
                    bytes_data = self._get_default_cfg_bytes()
                    self.shm.buf[:self._shm_buf_bytes_size] = bytes_data
                    logger.info("Rank %d shared memory is created.", self._rank_id)
                    break
                except Exception as ex:
                    # other process will go to step 1 and open shm file
                    try_times -= 1
                    logger.warning("Rank %d shared memory create failed, retry times = %d, %s has occur.",
                                   self._rank_id, try_times, str(ex))
                    time.sleep(random.uniform(0, 0.02))  # sleep 0 ~ 20 ms
        if try_times <= 0:
            raise RuntimeError("Failed to create shared memory.")

    def clean_resource(self):
        if sys.version_info >= (3, 8):
            self._clean_shm_over_py38()
        else:
            self._clean_shm_py37()

    def _clean_shm_over_py38(self):
        """Clean resource py38"""

        # clear shared memory
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
                logger.info("Rank %s unlink shm", self._rank_id)
            except Exception as ex:
                logger.warning("Rank %s unlink shm failed, may be removed, %s hs occur", self._rank_id, str(ex))
            self.shm = None

    def _clean_shm_py37(self):
        # clear shared memory
        if self.shm:
            try:
                self.shm.close()
                if self._memory_mapped_file and not self._memory_mapped_file.closed:
                    self._memory_mapped_file.close()
                elif self.fd:
                    os.close(self.fd)
                logger.info("Rank %s unlink shm", self._rank_id)
            except Exception as ex:
                logger.warning("Rank %s unlink shm failed, may be removed, %s has occur ", self._rank_id, str(ex))
            PathManager.remove_path_safety(os.path.dirname(self.shm_path))
            self.shm = None

    def _create_shm_py37(self):
        """Create a json monitor process based on whether the SharedMemory is successfully created py37"""
        logger.warning("Dynamic profiler is not work well on python 3.7x, "
                       "please update to python 3.8+ for better performance.")
        try_times = 10
        while try_times:
            try:
                # Step 1: try to open fd, first time fd not exists.
                self.fd = os.open(self.shm_path, os.O_EXCL | os.O_RDWR, stat.S_IRUSR | stat.S_IWGRP | stat.S_IRGRP)
                self._memory_mapped_file = os.fdopen(self.fd, 'rb')
                self.shm = mmap.mmap(self._memory_mapped_file.fileno(), length=self._shm_buf_bytes_size)
                self.is_create_process = False
                logger.info("Rank %d shared memory is connected.", self._rank_id)
                break
            except ValueError:
                time.sleep(0.02)
            except FileNotFoundError:
                try:
                    # Step 2: only one process can create fd successfully.
                    # Init mmap file need to write data
                    basedir = os.path.dirname(self.shm_path)
                    if not os.path.exists(basedir):
                        os.makedirs(basedir, exist_ok=True)
                    byte_data = self._get_default_cfg_bytes()
                    fd = os.open(self.shm_path,
                                 os.O_CREAT | os.O_EXCL | os.O_RDWR, stat.S_IRUSR | stat.S_IWGRP | stat.S_IRGRP)
                    with os.fdopen(fd, 'wb') as f:
                        f.write(byte_data)

                    # create mmap
                    self.fd = os.open(self.shm_path, os.O_EXCL | os.O_RDWR, stat.S_IRUSR | stat.S_IWGRP | stat.S_IRGRP)
                    self._memory_mapped_file = os.fdopen(self.fd, 'rb')
                    self.shm = mmap.mmap(self._memory_mapped_file.fileno(), length=self._shm_buf_bytes_size)
                    self.is_create_process = True
                    logger.info("Rank %d shared memory is created.", self._rank_id)
                    break
                except Exception as ex:
                    # other process will go to step 1 and open shm file
                    try_times -= 1
                    logger.warning("Rank %d shared memory create failed, retry times = %d, %s has occur .",
                                   self._rank_id, try_times, str(ex))
                    time.sleep(random.uniform(0, 0.02))  # sleep 0 ~ 20 ms

        if try_times <= 0:
            raise RuntimeError("Failed to create shared memory.")

    def read_bytes(self, read_time=False):
        """Read bytes from shared memory"""
        if sys.version_info >= (3, 8):
            if read_time:
                res = self.shm.buf[:self._time_bytes_size]
                return res
            else:
                res = self.shm.buf[self._time_bytes_size:self._shm_buf_bytes_size]
        else:
            self.shm.seek(0)
            if read_time:
                res = self.shm[:self._time_bytes_size]
                return res
            else:
                res = self.shm[self._time_bytes_size:self._shm_buf_bytes_size]
        res = bytes(res)
        bytes_data = res.lstrip()

        return bytes_data

    def write_bytes_over_py38(self, bytes_data: bytes):
        """Write bytes to shared memory"""
        bytes_data = bytes_data.ljust(self._shm_buf_bytes_size)
        self.shm.buf[:self._shm_buf_bytes_size] = bytes_data

    @staticmethod
    def write_bytes_py37(shm: mmap.mmap, bytes_data: bytes, buffer_size: int):
        """Write bytes to shared memory"""
        bytes_data = bytes_data.ljust(buffer_size)
        shm.seek(0)
        shm[:buffer_size] = bytes_data
