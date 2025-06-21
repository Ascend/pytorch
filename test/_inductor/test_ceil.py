# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import OperatorType, TestUtils
import torch_npu


class TestRelu(TestUtils):

    def op_calc(self, first_element):
        result = torch.ceil(first_element)
        return result

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float32', 'float16', 'bfloat16'])
    def test_pointwise_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element)

        torch.testing.assert_close(std_result, inductor_result)

instantiate_parametrized_tests(TestRelu)

if __name__ == "__main__":
    run_tests()
