# Copyright (c) 2024, Huawei Technologies.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
import struct

from torch_npu._C._profiler import (
    _get_syscnt_enable,
    _get_syscnt,
    _get_monotonic
)
from ..utils._error_code import ErrCode, prof_error
from ._profiler_path_creator import ProfPathCreator
from .analysis.prof_common_func._constant import Constant, print_error_msg, print_warn_msg
from .analysis.prof_common_func._file_manager import FileManager
from .analysis.prof_common_func._path_manager import ProfilerPathManager

__all__ = []


class ProfGCDetector:

    SAVE_FILE = "torch.gc_record"
    START_PHASE = "start"
    STOP_PHASE = "stop"

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold * Constant.NS_TO_MS
        self.time_info = {}
        self.save_info = []
        self.get_cur_ts = _get_syscnt if _get_syscnt_enable() else _get_monotonic

    def start(self):
        if self.gc_callback not in gc.callbacks:
            gc.callbacks.append(self.gc_callback)

    def stop(self):
        self.time_info.clear()
        self.save()
        self.save_info = []
        if self.gc_callback in gc.callbacks:
            gc.callbacks.remove(self.gc_callback)

    def save(self):
        if not self.save_info:
            return
        byte_arr = bytearray()
        for item in self.save_info:
            byte_arr.extend(struct.pack(Constant.GC_RECORD_FORMAT, *item))
        framework_dir = ProfilerPathManager.get_fwk_path(ProfPathCreator().get_prof_dir())
        output_path = os.path.join(framework_dir, self.SAVE_FILE)
        try:
            FileManager.create_bin_file_by_path(output_path, bytes(byte_arr))
        except Exception:
            print_error_msg(f"Can't create file: {output_path}" + prof_error(ErrCode.SYSCALL))

    def gc_callback(self, phase, info):
        pid = os.getpid()
        time_ns = _get_monotonic()
        if phase == self.START_PHASE:
            self.time_info[pid] = (time_ns, self.get_cur_ts())
        elif phase == self.STOP_PHASE:
            if pid not in self.time_info:
                print_warn_msg("Invalid pid in GC detect.")
            elif time_ns - self.time_info[pid][0] >= self.threshold:
                gc_info = (pid, self.time_info[pid][1], self.get_cur_ts())
                self.save_info.append(gc_info)
        else:
            print_warn_msg("Invalid GC detect phase.")
