# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
import pytest
from testutils import OperatorType, TestUtils
import torch_npu



class TestWhere(TestUtils):

    def op_calc(self, condition, first_element, second_element):
        return torch.where(condition, first_element, second_element)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float16', 'float32', 'bfloat16', 'int32'])   
    def test_pointwise_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_tensor(shape, dtype)
        condition = self._generate_tensor(shape, 'bool')

        std_result = self.op_calc(condition, first_element, second_element)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(condition, first_element, second_element)

        torch.testing.assert_close(std_result, inductor_result)


instantiate_parametrized_tests(TestWhere)

if __name__ == "__main__":
    run_tests()
