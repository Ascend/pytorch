# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
from torch.random import manual_seed as torch_manual_seed, seed as torch_seed
import torch_npu


class LogLevel:
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


def manual_seed(seed: int) -> None:
    r"""Sets the seed for generating random numbers. Returns a
    `torch.Generator` object.

    Args:
        seed (int): The desired seed.
    """
    _seed = int(seed)
    if not torch_npu.npu._in_bad_fork:
        torch_npu.npu.manual_seed_all(_seed)

    return torch_manual_seed(_seed)


def seed():
    r"""Sets the seed for generating random numbers to a non-deterministic
    random number. Returns a 64 bit number used to seed the RNG.
    """
    _seed = torch_seed()
    if not torch_npu.npu._in_bad_fork:
        torch_npu.npu.manual_seed_all(_seed)

    return _seed


def _print_log(level: str, msg: str):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    pid = os.getpid()
    print(f"{current_time}({pid})-[{level}] {msg}")


def print_info_log(info_msg: str):
    _print_log(LogLevel.INFO, info_msg)


def print_warn_log(warn_msg: str):
    _print_log(LogLevel.WARNING, warn_msg)


def print_error_log(error_msg: str):
    _print_log(LogLevel.ERROR, error_msg)
