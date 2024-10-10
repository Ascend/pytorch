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
import warnings
import torch_npu._C


class mstx:
    @staticmethod
    def mark(message:str = ""):
        torch_npu._C._mark(message)

    @staticmethod
    def range_start(message: str, stream=None) -> int:
        if not message:
            warnings.warn("Invalid message for mstx.range_start func. Please input valid message string.")
            return 0
        if stream:
            if isinstance(stream, torch_npu.npu.streams.Stream):
                stream = stream.npu_stream
                return torch_npu._C._mstx._range_start(message, stream)
            else:
                warnings.warn("Invalid stream for mstx.range_start func. Please input valid stream.")
                return 0
        else:
            return torch_npu._C._mstx._range_start_on_host(message)

    @staticmethod
    def range_end(range_id: int):
        if not isinstance(range_id, int):
            warnings.warn("Invalid message for mstx.range_start func. Please input return value from mstx.range_start.")
            return
        torch_npu._C._mstx._range_end(range_id)

    @staticmethod
    def mstx_range(message: str, stream=None):
        def wrapper(func):
            def inner(*args, **kargs):
                range_id = mstx.range_start(message, stream)
                ret = func(*args, **kargs)
                mstx.range_end(range_id)
                return ret
            return inner
        return wrapper
