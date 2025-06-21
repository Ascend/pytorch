# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
import pytest
from testutils import OperatorType, TestUtils
import torch_npu



class TestAttnCp(TestUtils):

    shape = (8, 8, 256, 128)
    dim = -1

    def foo(self, a, b, c):
        y = a + b
        y = y.sum(self.dim)
        y = y.unsqueeze(self.dim)
        y = y.broadcast_to(self.shape) + b
        y = c + y.permute(0, 1, 3, 2)
        return y


    def test_pointwise_cases(self):
        a, b = [torch.randn(self.shape, dtype=torch.float32, device="npu") for _ in range(2)]
        d = torch.randn(self.shape, dtype=torch.float32, device="npu")
        c = d.permute(0, 1, 3, 2).contiguous()
        func = torch.compile(self.foo, backend="inductor")
        r = func(a, b, c)
        r1 = self.foo(a, b, c)
        torch.testing.assert_close(r, r1, rtol=1e-3, atol=1e-3)

instantiate_parametrized_tests(TestAttnCp)

if __name__ == "__main__":
    run_tests()
