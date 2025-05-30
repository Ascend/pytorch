# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
import pytest
from testutils import OperatorType, TestUtils
import torch_npu
import torch_npu._inductor




class TestAbs(TestUtils):
    __TIME_LIMIT = 100
    __OPTYPE = OperatorType.POINTWISE

    def op_calc(self, first_element):
        result = torch.abs(first_element)
        return result

    # 在连续测试场景下,测试结果不稳定,建议单独重测批量测试未通过的 case
    # 若需测试更多数据类型，将dtype后面的list改成 ProtoTestCase._test_dtypes即可
    # 对indexing开关情况的测试需要用外部参数--npu_indexing=True/False完成

    @parametrize('shape', [(1024, 32), (256, 8)])
    @parametrize('dtype', ['float16', 'float32', 'bfloat16'])
    def test_pointwise_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element)
        torch.testing.assert_close(std_result, inductor_result, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestAbs)

if __name__ == "__main__":
    run_tests()
