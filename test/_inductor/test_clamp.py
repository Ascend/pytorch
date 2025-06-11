# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import OperatorType, TestUtils
import torch_npu
import torch_npu._inductor


class TestClamp(TestUtils):

    def op_calc(self, input, min=None, max=None):
        return input.clamp(min, max)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float16', 'float32', 'bfloat16', 'int32', 'int64'])
    def test_pointwise_cases_minmax_is_tensor(self, shape, dtype):
        min = self._generate_tensor(shape, dtype)
        max = self._generate_tensor(shape, dtype)

        first_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element, min=min, max=max)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, min=min, max=max)

        torch.testing.assert_close(std_result, inductor_result)

    @parametrize('shape', [(1,)])
    @parametrize('dtype', ['float32'])
    def test_pointwise_cases_single_scalar(self, shape, dtype):
        min = 0
        max = 100

        first_element = 200 * torch.rand(size=shape, dtype=eval('torch.' + dtype), device=torch.device("npu"))

        std_result = self.op_calc(first_element, min=min, max=max)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, min=min, max=max)
        torch.testing.assert_close(std_result, inductor_result)

    @parametrize('shape', [(1024, 32)])
    @parametrize('dtype', ['int32'])
    def test_pointwise_cases_minmax_is_number(self, shape, dtype):
        min = 0
        max = 100

        first_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element, min=min, max=max)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, min=min, max=max)

        torch.testing.assert_close(std_result, inductor_result)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float16', 'float32', 'bfloat16', 'int32', 'int64'])
    def test_pointwise_cases_max_only(self, shape, dtype):
        max = 100

        first_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element, min=None, max=max)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, min=None, max=max)

        torch.testing.assert_close(std_result, inductor_result)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float16', 'float32', 'bfloat16', 'int32', 'int64'])  
    def test_pointwise_cases_min_only(self, shape, dtype):
        min = 0

        first_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element, min=min, max=None)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, min=min, max=None)

        torch.testing.assert_close(std_result, inductor_result)

instantiate_parametrized_tests(TestClamp)

if __name__ == "__main__":
    run_tests()
