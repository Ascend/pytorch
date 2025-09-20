import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestEmbeddingDense(TestUtils):
    def op_calc(self, arg, embedding):
        output = embedding(arg)
        return output

    # UT skip, reason: precision fail
    # Added to pytorch-disable-tests.json
    def test_pointwise_cases(self):
        
        arg0 = torch.tensor([[14, 1, 2, 10, 0, 10, 0],
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
        embedding = nn.Embedding(16, 128).npu()
        std_sub = self.op_calc(arg0, embedding)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_sum = compiled_op_calc(arg0, embedding)
        self.assertEqual(std_sub, inductor_sum)


instantiate_parametrized_tests(TestEmbeddingDense)

if __name__ == "__main__":
    run_tests()
