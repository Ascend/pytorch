# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
import pytest
from testutils import OperatorType, TestUtils
import torch_npu
import torch_npu._inductor



class TestArgmax(TestUtils):

    def argmax(self, a, dim):
        return torch.argmax(a, dim)

    def test_argmax(self):
        shape = (512, 64)
        dim = -1
        a = torch.randn(shape, requires_grad=False, dtype=torch.float32, device='npu')

        argmax_triton = torch.compile(self.argmax, backend="inductor", dynamic=False)
        r = self.argmax(a, dim)
        r1 = argmax_triton(a, dim)
        torch.testing.assert_close(r, r1, rtol=1e-3, atol=1e-3)

instantiate_parametrized_tests(TestArgmax)

if __name__ == "__main__":
    run_tests()
