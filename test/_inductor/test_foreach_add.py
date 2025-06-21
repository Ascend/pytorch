# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import OperatorType, TestUtils
import torch_npu


class TestForeachAdd(TestUtils):

    def op_calc(self, first_element, second_element):
        tensor_list = [first_element, second_element]

        add_list = [first_element, second_element]
        result = torch._foreach_add_(tensor_list, add_list)
        return result

    # UT skip, reason: compile error, torch npu segmet fault
    # Added to pytorch-disable-tests.json
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['int32'])
    def test_pointwise_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element, second_element)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, second_element)

        torch.testing.assert_close(std_result, inductor_result, rtol=1e-1, atol=1e-1)

instantiate_parametrized_tests(TestForeachAdd)

if __name__ == "__main__":
    run_tests()
