# Copyright (c) 2023, Huawei Technologies.
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

import os
import time


class Constant:
    INVALID_VALUE = -1

    # dir name
    FRAMEWORK_DIR = "FRAMEWORK"
    OUTPUT_DIR = "ASCEND_PROFILER_OUTPUT"

    # file authority
    FILE_AUTHORITY = 0o640
    DIR_AUTHORITY = 0o750
    MAX_FILE_SIZE = 1024 * 1024 * 1024
    MAX_CSV_SIZE = 1024 * 1024 * 1024

    # tlv constant struct
    CONSTANT_BYTES = "constant_bytes"

    # field name
    SEQUENCE_UNMBER = "Sequence number"
    FORWORD_THREAD_ID = "Fwd thread id"
    OP_NAME = "op_name"
    INPUT_SHAPES = "Input Dims"
    INPUT_DTYPES = "Input type"
    CALL_STACK = "Call stack"
    MODULE_HIERARCHY = "Module Hierarchy"
    FLOPS = "flops"
    NAME = "name"

    # torch op acl relation field name
    ACL_START_TIME = "acl_start_time"

    # trace constant
    PROCESS_NAME = "process_name"
    PROCESS_LABEL = "process_labels"
    PROCESS_SORT = "process_sort_index"
    THREAD_NAME = "thread_name"
    THREAD_SORT = "thread_sort_index"

    # framework
    TENSORBOARD_TRACE_HABDLER = "tensorboard_trace_handler"
    EXPORT_CHROME_TRACE = "export_chrome_trace"

    ACL_OP_EXE_NAME = ("AscendCL@aclopCompileAndExecute", "AscendCL@aclopCompileAndExecuteV2")
    NPU_PID = "3_0"


def print_error(file_name: str, msg: str) -> None:
    print_msg = time.strftime("%a %d %b %Y %H:%M:%S", time.localtime()) + f"[ERROR] [{os.getpid()}] {file_name}: " + msg
    print(print_msg)


def print_warn(msg: str) -> None:
    print_msg = time.strftime("%a %d %b %Y %H:%M:%S", time.localtime()) + f"[WARN] [{os.getpid()}] profiler.py: " + msg
    print(print_msg)
