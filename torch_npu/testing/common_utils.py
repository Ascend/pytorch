from functools import wraps
from itertools import product
from contextlib import contextmanager

import os
import sys
import unittest
import json
import stat
import atexit
import threading
import tempfile
import torch
import numpy as np

from torch.testing._internal.common_utils import TEST_MKL

import torch_npu

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

def skipIfUnsupportMultiNPU(npu_number_needed):
    def skip_dec(func):
        def wrapper(self):
            if not torch.npu.is_available() or torch.npu.device_count() < npu_number_needed:
                raise unittest.SkipTest("Multi-NPU condition not satisfied")
            return func(self)
        return wrapper
    return skip_dec

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

PERF_TEST_ENABLE = (os.getenv('PERF_TEST_ENABLE', default='').upper() in ['ON', '1', 'YES', 'TRUE', 'Y'])
PERF_BASELINE_FILE = os.getenv("PERF_BASELINE_FILE", default=os.path.join(os.getcwd(), "performance_baseline.json"))

class Baseline(object):

    def __init__(self, baselineFile):
        self._baseline = {}
        self._baselineFile = baselineFile
        self._mutex = threading.Lock()
        if os.path.exists(self._baselineFile):
            with open(self._baselineFile, "r") as f:
                self._baseline = json.load(f)

    def get_baseline(self, resourceId):
        return self._baseline.get(resourceId)

    def set_baseline(self, resourceId, baseline):
        with self._mutex:
            self._baseline[resourceId] = baseline

    def save_baseline(self):
        with self._mutex:
            with os.fdopen(os.open(self._baselineFile, os.O_RDWR|os.O_CREAT, stat.S_IWUSR|stat.S_IRUSR), "w") as f:
                json.dump(self._baseline, f)

PerfBaseline = Baseline(PERF_BASELINE_FILE)

@atexit.register
def dump_baseline():
    if PERF_TEST_ENABLE:
        PerfBaseline.save_baseline()
