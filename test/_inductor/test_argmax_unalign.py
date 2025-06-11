# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
import pytest
from testutils import OperatorType, TestUtils
import torch_npu
import torch_npu._inductor

torch_npu._inductor.config.enable_npu_indexing = True


class TestMaxWithIndex(TestUtils):
    def op_calc(self, input_element, dim):
        return torch.argmax(input_element, dim)

    @parametrize('shape', [(512, 64)]) # (513, 64), (514,33)
    @parametrize('dim', [-1])
    @parametrize('dtype', ['float32'])
    def test_reduction_cases(self, shape, dim, dtype):
        print('npu_indexing= {}'.format(torch_npu._inductor.config.enable_npu_indexing))
        input_element = torch.randn(size=shape, dtype=eval('torch.' + dtype), device=torch.device("npu")) * 2000
        std_argmax = self.op_calc(input_element, dim)
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor", dynamic=False)
        inductor_argmax = compiled_op_calc(input_element, dim)
        torch.testing.assert_close(std_argmax, inductor_argmax, rtol=1e-2, atol=1e-2)

instantiate_parametrized_tests(TestMaxWithIndex)

if __name__ == "__main__":
    run_tests()
