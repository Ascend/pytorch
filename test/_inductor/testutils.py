# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from enum import Enum, unique
from collections.abc import Sequence
import os
import time
import numpy as np
import torch
from torch.testing._internal.common_utils import TestCase
import torch_npu


@unique
class OperatorType(Enum):
    POINTWISE = 'POINTWISE'
    REDUCTION = 'REDUCTION'


class TestUtils(TestCase):
    _pointwise_test_shape2d = [(4096, 256), (1024, 32), (8, 2048), (8, 4096)]  # (8, 4), (8, 8), not supported
    _pointwise_test_shape3d = [(8, 8, 4), (8, 8, 8), (8, 8, 2048), (8, 8, 4096)]
    _pointwise_test_shape4d = [(128, 128, 4096, 4), (128, 128, 4096, 8),
                               (32, 32, 1024, 1024)]  # 128*128*4096*2048 is too big(512G)
    _pointwise_test_shapes = _pointwise_test_shape2d + _pointwise_test_shape3d + _pointwise_test_shape4d

    _pointwise_demo_shapes = [(1024, 32), (8, 16, 256, 32)]
    _reduction_extest_shape4d = [(8, 8, 8, 16384), (8, 8, 16384, 8), (8, 16384, 8, 8), (16384, 8, 8, 8)]
    _reduction_extest_dim4d = [-1, -2, 1, 0]
    _reduction_extest_SDbinding = list(zip(_reduction_extest_shape4d, _reduction_extest_dim4d))

    _test_dtypes = ['float32', 'int32', 'float16', 'bfloat16', 'int64']

    @staticmethod
    def _generate_tensor(shape, dtype, floatPOSIFLAG=0):
        if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
            if floatPOSIFLAG:
                return 1000 * torch.rand(size=shape, dtype=eval('torch.' + dtype), device=torch.device("npu"))
            else:
                return torch.randn(size=shape, dtype=eval('torch.' + dtype), device=torch.device("npu")) * 2000
        elif dtype == 'int32' or dtype == 'int64':
            return torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype), device=torch.device("npu"))
        elif dtype == 'bool':
            return torch.randint(low=0, high=2, size=shape, device=torch.device("npu")).bool()
        else:
            raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


def benchmark_test(fn, fn_triton, args=(), name="gen_fn", times=10, repeat=10, profile=False):
    print(f"--------------------benchmark_{name} for {times * repeat} times--------------------")
    stream = torch.npu.current_stream()
    profiler = None
    if profile:
        profiler = create_profiler()

    stream.synchronize()
    if profile:
        profiler.start()
    start = time.perf_counter()
    for _ in range(times * repeat):
        fn_triton(*args)
        if profile:
            profiler.step()
    stream.synchronize()
    end = time.perf_counter()
    if profile:
        profiler.stop()
    time_compiled = (end - start) / (times * repeat)
    time_compiled *= 1000000
    print(f"time_compiled:{time_compiled:.6f}")

    if profile:
        profiler = create_profiler()
    print(f"Runing eager {name} for {times * repeat} times")
    start = time.perf_counter()
    for _ in range(times * repeat):
        fn(*args)
        if profile:
            profiler.step()
    stream.synchronize()
    end = time.perf_counter()
    time_eager = (end - start) / (times * repeat)
    time_eager *= 1000000
    print(f"time_eager:{time_eager:.6f}")
    accelerated = (time_eager - time_compiled) / time_compiled * 100
    print(f"Accelerated: {accelerated:.4f}% eager takes {time_eager:.3f} us, triton takes {time_compiled:.3f} us")

    return time_eager, time_compiled
