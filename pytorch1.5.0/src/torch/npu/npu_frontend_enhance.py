# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
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

import torch._C
import os
import contextlib
# this file is used to enhance the npu frontend API by set_option or other.

__all__ = ["set_option", "set_dump", "init_dump", "finalize_dump", "global_step_inc", "set_start_fuzz_compile_step", 
           "iteration_start", "iteration_end", "profile"]

def set_option(option):
    if not isinstance(option, dict):
        raise TypeError("npu option must be a dict.")

    if option.get("MM_BMM_ND_ENABLE") is "enable":
        torch.npu.set_mm_bmm_format_nd(True)

    for option_name, option_value in option.items():
        option[option_name] = str(option_value)
    torch._C._npu_setOption(option)

def init_dump():
    option = {"mdldumpswitch":"enable"}
    torch._C._npu_setOption(option)

def set_dump(cfg_file):
    if not os.path.exists(cfg_file):
        raise AssertionError("cfg_file %s path not exists."%(cfg_file))
    cfg_file = os.path.abspath(cfg_file)
    option = {"mdldumpconfigpath": cfg_file}
    torch._C._npu_setOption(option)

def finalize_dump():
    option = {"mdldumpswitch": "disable"}
    torch._C._npu_setOption(option)

def iteration_start():
    option = {"deliverswitch": "enable"}
    torch._C._npu_setOption(option)

def iteration_end():
    option = {"deliverswitch": "disable"}
    torch._C._npu_setOption(option)

_GLOBAL_STEP=0
_START_FUZZ_COMPILE_STEP=1
def global_step_inc():
    global _GLOBAL_STEP
    _GLOBAL_STEP += 1

    option = {"fuzzycompileswitch": "enable" if _GLOBAL_STEP >= _START_FUZZ_COMPILE_STEP \
        else "disable"}
    torch._C._npu_setOption(option)

def set_start_fuzz_compile_step(step):
    if not isinstance(step, int):
        raise TypeError("step must be a int, but got ", type(step))
    
    global _START_FUZZ_COMPILE_STEP
    _START_FUZZ_COMPILE_STEP = step
    option = {"fuzzycompileswitch": "disable"}
    torch._C._npu_setOption(option)

def prof_init(path):
    if not os.path.exists(path):
        raise AssertionError("profiler_result_path: %s not exists."%(path))
    profiler_result_path = os.path.abspath(path)
    option = {"profilerResultPath": profiler_result_path}
    torch._C._npu_setOption(option)

def prof_start():
    option = {"profiling": "start"}
    torch._C._npu_setOption(option)

def prof_stop():
    option = {"profiling": "stop"}
    torch._C._npu_setOption(option)

def prof_finalize():
    option = {"profiling": "finalize"}
    torch._C._npu_setOption(option)

@contextlib.contextmanager
def profile(profiler_result_path):
    try:
        prof_init(profiler_result_path)
        prof_start()
        yield "profile"
    finally:
        prof_stop()
        prof_finalize()
