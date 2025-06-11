# -*- coding: utf-8 -*-
# Copyright (c) Huawei TechNologies Co., Ltd. 2023-2023. All rights reserved.
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import OperatorType, TestUtils
import torch_npu
import torch_npu._inductor


class TestEmbeddingDense():

    def op_calc(self, input):
        embedding = nn.Embedding(16, 128).npu()
        output = embedding(input)
        return output

    def test_pointwise_cases(self):
        torch_npu._inductor.config.enable_npu_indexing = True

        input = torch.tensor([[14, 1, 2, 10, 0, 10, 0],
                        [9, 13, 13, 4, 7, 15, 14],
                        [8, 0, 3, 15, 4, 2, 6],
                        [15, 12, 13, 9, 0, 8, 1],
                        [8, 15, 4, 15, 12, 9, 3],
                        [6, 11, 12, 8, 0, 13, 8],
                        [4, 10, 1, 12, 0, 0, 4],
                        [6, 6, 15, 6, 0, 10, 15],
                        [2, 5, 14, 0, 5, 7, 9],
                        [13, 4, 14, 11, 11, 9, 2],
                        [1, 1, 5, 1, 1, 6, 14],
                        [3, 9, 8, 4, 13, 8, 3],
                        [4, 10, 8, 13, 6, 8, 3]], device='npu:0')

        std_sub = self.op_calc(input)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_sum = compiled_op_calc(input)
        torch.testing.assert_close(std_sub, inductor_sum)

instantiate_parametrized_tests(TestEmbeddingDense)

if __name__ == "__main__":
    run_tests()
