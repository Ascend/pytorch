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

from functools import wraps
from itertools import product
from contextlib import contextmanager

import os
import sys
import unittest
import tempfile
import torch
import torch_npu
import numpy as np

from torch.testing._internal.common_utils import TEST_MKL


IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    @contextmanager
    def TemporaryFileName():
        # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
        # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
        # close the file after creation and try to remove it manually
        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            f.close()
            yield f.name
        finally:
            os.unlink(f.name)
else:
    @contextmanager  # noqa: T484
    def TemporaryFileName():
        with tempfile.NamedTemporaryFile() as f:
            yield f.name


@contextmanager
def freeze_rng_state():
    rng_state = torch.get_rng_state()
    yield
    torch.set_rng_state(rng_state)


def iter_indices(tensor):
    if tensor.dim() == 0:
        return range(0)
    if tensor.dim() == 1:
        return range(tensor.size(0))
    return product(*(range(s) for s in tensor.size()))


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def set_npu_device():
    npu_device = get_npu_device()
    torch.npu.set_device(npu_device)
    print(f"Your device is {npu_device}")
    return npu_device


def get_npu_device():
    npu_device = os.environ.get('SET_NPU_DEVICE')
    if npu_device is None:
        npu_device = "npu:0"
    else:
        npu_device = f"npu:{npu_device}"
    return npu_device


def create_common_tensor(item, minValue, maxValue, device=None):
    if device is None:
        device = get_npu_device()
        
    dtype = item[0]
    npu_format = item[1]
    shape = item[2]
    input1 = np.random.uniform(minValue, maxValue, shape).astype(dtype)
    cpu_input = torch.from_numpy(input1)
    npu_input = torch.from_numpy(input1).to(device)
    if npu_format != -1:
        npu_input = torch_npu.npu_format_cast(npu_input, npu_format)
    return cpu_input, npu_input

def compare_res_new(cpu_output, npu_output, testcase_name):
    if cpu_output.shape != npu_output.shape:
        return print("result shape error!", cpu_output.shape, npu_output.shape)
    if cpu_output.dtype != npu_output.dtype:
        return print("result dtype error!", cpu_output.dtype, npu_output.dtype)
    if cpu_output.dtype == np.int32:
        result = np.equal(cpu_output, npu_output)
        if result is False:
            return print('testcase_name={0}, npu datatype={1} shape={2} fails!'.format(
                testcase_name, npu_output.dtype, npu_output.shape))
    elif cpu_output.dtype == np.float16:
        result = np.allclose(npu_output, cpu_output, 0.0001, 0)
        if result is False:
            return print('testcase_name={0}, npu datatype={1} shape={2} fails!'.format(
                testcase_name, npu_output.dtype, npu_output.shape))
    elif cpu_output.dtype == np.float32:
        result = np.allclose(npu_output, cpu_output, 0.0001, 0)
        print(npu_output, cpu_output)
        print(result)
        if not result:
            return print('testcase_name={0}, npu datatype={1} shape={2} fails!'.format(
                testcase_name, npu_output.dtype, npu_output.shape))
    print('testcase_name={0}, datatype={1} shape={2} pass!'.format(testcase_name, cpu_output.dtype, cpu_output.shape))


def __generate_2args_broadcast_cases(device=None):
    if device is None:
        device = get_npu_device()
        
    # Set broadcast and no axis, i.e. broadcasting 1.
    X = np.random.rand(2, 3, 4, 5).astype(np.float32)
    Y = np.random.rand(1, 1, 1).astype(np.float32)

    cpu_x = torch.from_numpy(X)
    npu_x = torch.from_numpy(X).to(device)

    cpu_y = torch.from_numpy(Y)
    npu_y = torch.from_numpy(Y).to(device)

    yield cpu_x, cpu_y, npu_x, npu_y

    # broadcasting last two dimensions.
    X = np.random.rand(2, 3, 4, 5).astype(np.float32)
    Y = np.random.rand(4, 5).astype(np.float32)

    cpu_x = torch.from_numpy(X)
    npu_x = torch.from_numpy(X).to(device)

    cpu_y = torch.from_numpy(Y)
    npu_y = torch.from_numpy(Y).to(device)

    yield cpu_x, cpu_y, npu_x, npu_y

def test_2args_broadcast(fn):
    output_list = []
    for cpu_x, cpu_y, npu_x, npu_y in __generate_2args_broadcast_cases():
        cpu_out = fn(cpu_x, cpu_y).numpy()
        npu_out = fn(npu_x, npu_y).to("cpu").numpy()
        output_list.append([cpu_out, npu_out])

    return output_list


def create_dtype_tensor(shape, dtype, npu_format=-1, min_value=-5, max_value=5, no_zero=False, device=None):
    if device is None:
        device = get_npu_device()
        
    if dtype == torch.bool:
        x = np.random.randint(0, 2, size=shape).astype(bool)

    elif dtype == torch.half:
        x = np.random.uniform(min_value, max_value, shape).astype(np.float16)
    
    elif dtype == torch.float:
        x = np.random.uniform(min_value, max_value, shape).astype(np.float32)

    else:
        x = np.random.randint(min_value, max_value+1, size = shape).astype(np.int32)

    if no_zero:
        ones = np.ones_like(x)
        x = np.where(x != 0, x, ones)

    cpu_input = torch.from_numpy(x)
    npu_input = torch.from_numpy(x).to(device)
    if npu_format != -1 and (dtype in [torch.float, torch.half]):
        npu_input = torch_npu.npu_format_cast(npu_input, npu_format)
    return cpu_input, npu_input


def check_operators_in_prof(expected_operators, prof, unexpected_operators=None):
    unexpected_operators = unexpected_operators or []
    prof_key_averages = prof.key_averages()
    if not prof_key_averages:
        return print("torch profiling is empty, please check it")
    for prof_item in prof_key_averages:        
        if prof_item.key in unexpected_operators:
            # if unexpected oprators are called, pattern inferring in trans-contiguous is failed
            return False
        elif prof_item.key in expected_operators:
            # if expected oprator is called, empty it in expected_operators list
            expected_operators.remove(prof_item.key)
            
    # if expected_operators list is empty, all oprators have been called
    if not expected_operators:
        return True
    return False


# Decorator that skips a test if the given condition is true.
# Notes:
#   (1) Skip conditions stack.
#   (2) Skip conditions can be bools or strings. If a string the
#       test base must have defined the corresponding attribute to be False
#       for the test to run. If you want to use a string argument you should
#       probably define a new decorator instead (see below).
#   (3) Prefer the existing decorators to defining the 'device_type' kwarg.
class SkipIf(object):

    def __init__(self, dep, reason, device_type=None):
        self.dep = dep
        self.reason = reason
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def dep_fn(slf, device, *args, **kwargs):
            if self.device_type is None or self.device_type == slf.device_type:
                if ((isinstance(self.dep, str) and getattr(slf, self.dep, True))
                    or (isinstance(self.dep, bool) and self.dep)):
                    raise unittest.SkipTest(self.reason)

            return fn(slf, device, *args, **kwargs)
        return dep_fn


# Skips a test on CPU if the condition is true.
class SkipCPUIf(SkipIf):

    def __init__(self, dep, reason):
        super(SkipCPUIf, self).__init__(dep, reason, device_type='cpu')


class ExpectedFailure(object):

    def __init__(self, device_type):
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def efail_fn(slf, device, *args, **kwargs):
            if self.device_type is None or self.device_type == slf.device_type:
                try:
                    fn(slf, device, *args, **kwargs)
                except Exception:
                    return
                else:
                    slf.fail('expected test to fail, but it passed')

            return fn(slf, device, *args, **kwargs)
        return efail_fn


class OnlyOn(object):

    def __init__(self, device_type):
        self.device_type = device_type

    def __call__(self, fn):

        @wraps(fn)
        def only_fn(slf, device, *args, **kwargs):
            if self.device_type != slf.device_type:
                reason = "Only runs on {0}".format(self.device_type)
                raise unittest.SkipTest(reason)

            return fn(slf, device, *args, **kwargs)

        return only_fn


# Decorator that provides all available devices of the device type to the test
# as a list of strings instead of providing a single device string.
# Skips the test if the number of available devices of the variant's device
# type is less than the 'num_required_devices' arg.
class DeviceCountAtLeast(object):

    def __init__(self, num_required_devices):
        self.num_required_devices = num_required_devices

    def __call__(self, fn):
        assert not hasattr(fn, 'num_required_devices'), "DeviceCountAtLeast redefinition for {0}".format(fn.__name__)
        fn.num_required_devices = self.num_required_devices

        @wraps(fn)
        def multi_fn(slf, devices, *args, **kwargs):
            if len(devices) < self.num_required_devices:
                reason = "fewer than {0} devices detected".format(self.num_required_devices)
                raise unittest.SkipTest(reason)

            return fn(slf, devices, *args, **kwargs)

        return multi_fn


# Skips a test on CPU if LAPACK is not available.
class SkipCPUIfNoLapack(object):

    def __call__(self, fn):
        return SkipCPUIf(not torch._C.has_lapack, "PyTorch compiled without Lapack")(fn)


# Skips a test on CPU if MKL is not available.
class SkipCPUIfNoMkl(object):

    def __call__(fn):
        return SkipCPUIf(not TEST_MKL, "PyTorch is built without MKL support")(fn)


class SkipIfNoLapack(object):

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not torch._C.has_lapack:
                raise unittest.SkipTest('PyTorch compiled without Lapack')
            else:
                fn(*args, **kwargs)
        return wrapper


class SkipIfNotRegistered(object):
    """Wraps the decorator to hide the import of the `core`.

    Args:
        op_name: Check if this op is registered in `core._REGISTERED_OPERATORS`.
        message: message to fail with.

    Usage:
        @SkipIfNotRegistered('MyOp', 'MyOp is not linked!')
            This will check if 'MyOp' is in the caffe2.python.core
    """
    def __call__(op_name, message):
        try:
            from caffe2.python import core
            skipper = unittest.skipIf(op_name not in core._REGISTERED_OPERATORS, message)
        except ImportError:
            skipper = unittest.skip("Cannot import `caffe2.python.core`")
        return skipper