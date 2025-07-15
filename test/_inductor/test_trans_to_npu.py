import pytest
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestTransToNpu(TestUtils):
    def op_add(self, a, b):
        return a + b

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float16', 'float32'])
    def test_trans_to_npu(self, shape, dtype):
        from torch_npu.contrib import transfer_to_npu

        input_element1 = self._generate_tensor(shape, dtype)
        input_element2 = self._generate_tensor(shape, dtype)

        std_result = self.op_add(input_element1, input_element2)

        compiled_op_add = torch.compile(self.op_add, backend="inductor", dynamic=False)
        inductor_result1 = compiled_op_add(input_element1, input_element2)
        torch.testing.assert_close(std_result, inductor_result1, atol=1e-3, rtol=1e-3)

instantiate_parametrized_tests(TestTransToNpu)

if __name__ == "__main__":
    run_tests()