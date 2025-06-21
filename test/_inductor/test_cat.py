# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import OperatorType, TestUtils
import torch_npu


class TestCat(TestUtils):

    def op_calc(self, input_element, dim):
        return torch.cat([input_element, input_element], dim)

    # case：change shapes
    @parametrize('shape', [(8, 16, 32, 64)])
    @parametrize('dim', [-1])
    @parametrize('dtype', ['bfloat16'])
    def test_reduction_cases_shapes(self, shape, dim, dtype):
        input_element = self._generate_tensor(shape, dtype)
        std_cat = self.op_calc(input_element, dim)
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_cat = compiled_op_calc(input_element, dim)
        rtol = 1e-1
        atol = 1e-1
        torch.testing.assert_close(std_cat, inductor_cat, equal_nan=True, rtol=rtol, atol=atol)

instantiate_parametrized_tests(TestCat)

if __name__ == "__main__":
    run_tests()
