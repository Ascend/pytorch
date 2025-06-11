# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import OperatorType, TestUtils
import torch_npu
import torch_npu._inductor


class TestVarMean(TestUtils):
    
    def op_calc(self, input_element, dim):
        return torch.var_mean(input_element, dim)

    # case：The shape must not be too large
    @parametrize('shape', [(8, 64, 128)])
    @parametrize('dim', [0, 1, 2, (0, 2), (0, 1)])
    @parametrize('dtype', ['float32'])
    def test_reduction_cases_shapes(self, shape, dim, dtype):

        input_element = self._generate_tensor(shape, dtype)

        std_var, std_mean = self.op_calc(input_element, dim)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor", dynamic=False)
        inductor_var, inductor_mean = compiled_op_calc(input_element, dim)

        rtol = 1e-1
        atol = 1e-1
        torch.testing.assert_close(std_var, inductor_var, equal_nan=True, rtol=rtol, atol=atol)
        torch.testing.assert_close(std_mean, inductor_mean, equal_nan=True, rtol=rtol, atol=atol)


instantiate_parametrized_tests(TestVarMean)

if __name__ == "__main__":
    run_tests()
