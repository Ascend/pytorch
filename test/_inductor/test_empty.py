# -*- coding: utf-8 -*-
# Copyright (c) Huawei TechNologies Co., Ltd. 2023-2023. All rights reserved.
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import OperatorType, TestUtils
import torch_npu
import torch_npu._inductor


class TestEmpty(TestUtils):
    
    def op_calc(self):
        x = torch.empty(8, 64, 128, dtype=torch.float32).npu()
        x.uniform_(-100, 100)
        return x

    def op_calc_empty_permuted(self):
        input_shape = (8, 64, 128)
        physical_layout = (0, 1, 2)
        x = torch.empty_permuted(input_shape, physical_layout).npu()
        x.uniform_(-100, 100)
        return x

    # case： change shapes
    @parametrize('shape', [(8, 64, 128)])
    @parametrize('dim', [0])
    @parametrize('dtype', ['float32'])
    def test_cases_empty(self, shape, dim, dtype):

        std_ret = self.op_calc()
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_ret = compiled_op_calc()

        self.assertTrue(inductor_ret.numel() > 0)

    @parametrize('shape', [(8, 64, 128)])
    @parametrize('dim', [0])
    @parametrize('dtype', ['float32'])
    def test_cases_empty_permuted(self, shape, dim, dtype):
        std_ret = self.op_calc_empty_permuted()
        compiled_op_calc = torch.compile(self.op_calc_empty_permuted, backend="inductor")
        inductor_ret = compiled_op_calc()

        self.assertTrue(inductor_ret.numel() > 0)


instantiate_parametrized_tests(TestEmpty)

if __name__ == "__main__":
    run_tests()

