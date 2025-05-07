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
import functools
import torch_npu._C
from torch_npu.utils.utils import _print_error_log

__all__ = ["mstx"]


def _no_exception_func(default_ret=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
            except Exception as ex:
                _print_error_log(f"Call {func.__name__} failed. Exception: {str(ex)}")
                return default_ret
            return result
        return wrapper
    return decorator


class mstx:
    @staticmethod
    @_no_exception_func()
    def mark(message: str, stream=None, domain: str = 'default'):
        if not message or not isinstance(message, str):
            warnings.warn("Invalid message for mstx.mark func. Please input valid message string.")
            return
        if not isinstance(domain, str):
            warnings.warn("Invalid domain for mstx.mark func. Please input valid domain string.")
            return
        if stream:
            if isinstance(stream, torch_npu.npu.streams.Stream):
                stream = stream.npu_stream
                torch_npu._C._mstx._mark(message, stream, domain)
            else:
                warnings.warn("Invalid stream for mstx.mark func. Please input valid stream.")
                return
        else:
            torch_npu._C._mstx._mark_on_host(message, domain)

    @staticmethod
    @_no_exception_func()
    def range_start(message: str, stream=None, domain: str = 'default') -> int:
        if not message or not isinstance(message, str):
            warnings.warn("Invalid message for mstx.range_start func. Please input valid message string.")
            return 0
        if not domain or not isinstance(domain, str):
            warnings.warn("Invalid domain for mstx.range_start func. Please input valid domain string.")
            return 0
        if stream:
            if isinstance(stream, torch_npu.npu.streams.Stream):
                stream = stream.npu_stream
                return torch_npu._C._mstx._range_start(message, stream, domain)
            else:
                warnings.warn("Invalid stream for mstx.range_start func. Please input valid stream.")
                return 0
        else:
            return torch_npu._C._mstx._range_start_on_host(message, domain)

    @staticmethod
    @_no_exception_func()
    def range_end(range_id: int, domain: str = 'default'):
        if not isinstance(range_id, int):
            warnings.warn("Invalid message for mstx.range_end func. Please input return value from mstx.range_start.")
            return
        if not domain or not isinstance(domain, str):
            warnings.warn("Invalid domain for mstx.range_end func. Please input valid domain string.")
            return
        torch_npu._C._mstx._range_end(range_id, domain)

    @staticmethod
    @_no_exception_func()
    def mstx_range(message: str, stream=None, domain: str = 'default'):
        def wrapper(func):
            def inner(*args, **kargs):
                range_id = mstx.range_start(message, stream, domain)
                ret = func(*args, **kargs)
                mstx.range_end(range_id, domain)
                return ret
            return inner
        return wrapper
