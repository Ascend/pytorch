import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestEmbeddingDenseBackward(TestUtils):
    def op_calc(self, slice_4, sum_23):
        result = torch.ops.aten.embedding_dense_backward.default(sum_23, slice_4, 512, -1, False)
        return result

    @parametrize('shape', [(1, 512, 128)])
    @parametrize('dtype', ['float32'])
    def test_pointwise_cases(self, shape, dtype):
        first_element = torch.randint(low=0, high=128, size=(1, 512), dtype=torch.int64).npu()
        second_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element, second_element)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, second_element)

        self.assertEqual(std_result, inductor_result, atol=1e-1, rtol=1e-1)


instantiate_parametrized_tests(TestEmbeddingDenseBackward)

if __name__ == "__main__":
    run_tests()
