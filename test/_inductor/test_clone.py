# -*- coding: utf-8 -*-
# Copyright (c) Huawei TechNologies Co., Ltd. 2023-2023. All rights reserved.
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import OperatorType, TestUtils
import torch_npu
import torch_npu._inductor


class TestClone(TestUtils):
    
    def op_calc(self, input_element, dim):
        return torch.clone(input_element)

    # case： change shapes
    @parametrize('shape', [(8, 64, 128)])
    @parametrize('dim', [0])
    @parametrize('dtype', ['float32'])
    def test_reduction_cases_shapes(self, shape, dim, dtype):
        input_element = self._generate_tensor(shape, dtype)
        std_ret = self.op_calc(input_element, dim)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_ret = compiled_op_calc(input_element, dim)

        torch.testing.assert_close(std_ret, inductor_ret, equal_nan=True)

instantiate_parametrized_tests(TestClone)

if __name__ == "__main__":
    run_tests()
