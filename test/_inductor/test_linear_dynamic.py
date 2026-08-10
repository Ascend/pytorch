import unittest
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestUnifiedAxis(TestUtils):

    def op_calc_dynamic(self, x, y, batch_size, seq_len, hidden1, hidden2, dim1, dim2):
        view_1 = x.view(batch_size, seq_len, hidden1, dim1).permute(0, 1, 3, 2).reshape(batch_size, seq_len, hidden1*dim1)
        view_3 = y.view(batch_size, seq_len, hidden2, dim2).permute(0, 1, 3, 2).reshape(batch_size, seq_len, hidden2*dim2)
        add = view_1 + view_3
        return add


    @parametrize('dtype', ['float32'])
    def test_symbol_cases_dynamic_true(self, dtype):
        compiled_op_calc = torch.compile(self.op_calc_dynamic, backend="inductor", dynamic=True)
        for batch in [256, 512]:
            for seq in [80, 160]:
                hidden1 = 160
                hidden2 = 160
                dim1 = 40          # permute 的最后一个维度
                dim2 = 40           # permute_1 的最后一个维度

                x = self._generate_tensor((batch, seq, hidden1, dim1), dtype)
                torch._dynamo.mark_dynamic(x, 0, min=256, max=512)
                torch._dynamo.mark_dynamic(x, 1, min=80, max=160)
                x = x.as_strided(
                    (batch, seq, hidden1, dim1),
                    (seq * hidden1 * dim1, hidden1 * dim1, dim1, 1)   # 故意构造非连续 stride
                )
                y = self._generate_tensor((batch, seq, hidden2, dim2), dtype)
                torch._dynamo.mark_dynamic(y, 0, min=256, max=512)
                torch._dynamo.mark_dynamic(y, 1, min=80, max=160)
                y = y.as_strided(
                    (batch, seq, hidden2, dim2),
                    (seq * hidden2 * dim2, hidden2 * dim2, dim2, 1)
                )
                std_result = self.op_calc_dynamic(x, y, batch, seq, hidden1, hidden2, dim1, dim2)
                inductor_result = compiled_op_calc(x, y, batch, seq, hidden1, hidden2, dim1, dim2)

                self.assertEqual(std_result, inductor_result, atol=1e-2, rtol=1e-2)


    @parametrize('dtype', ['float32'])
    def test_symbol_cases_dynamic_false(self, dtype):
        compiled_op_calc = torch.compile(self.op_calc_dynamic, backend="inductor", dynamic=True)
        for batch in [256, 512]:
            for seq in [80, 160]:
                hidden1 = 64
                hidden2 = 160
                dim1 = 40          # permute 的最后一个维度
                dim2 = 16           # permute_1 的最后一个维度

                x = self._generate_tensor((batch, seq, hidden1, dim1), dtype)
                torch._dynamo.mark_dynamic(x, 0, min=256, max=512)
                torch._dynamo.mark_dynamic(x, 1, min=80, max=160)
                x = x.as_strided(
                    (batch, seq, hidden1, dim1),
                    (seq * hidden1 * dim1, hidden1 * dim1, dim1, 1)   # 故意构造非连续 stride
                )
                y = self._generate_tensor((batch, seq, hidden2, dim2), dtype)
                torch._dynamo.mark_dynamic(y, 0, min=256, max=512)
                torch._dynamo.mark_dynamic(y, 1, min=80, max=160)
                y = y.as_strided(
                    (batch, seq, hidden2, dim2),
                    (seq * hidden2 * dim2, hidden2 * dim2, dim2, 1)
                )

                std_result = self.op_calc_dynamic(x, y, batch, seq, hidden1, hidden2, dim1, dim2)
                inductor_result = compiled_op_calc(x, y, batch, seq, hidden1, hidden2, dim1, dim2)

                self.assertEqual(std_result, inductor_result, atol=1e-2, rtol=1e-2)


instantiate_parametrized_tests(TestUnifiedAxis)


if __name__ == "__main__":
    run_tests()