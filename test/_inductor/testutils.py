from collections.abc import Sequence
import os
import time
import numpy as np
import torch
from torch.testing._internal.common_utils import TestCase
import torch_npu


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
