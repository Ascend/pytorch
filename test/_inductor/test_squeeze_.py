import torch
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils
import torch_npu


class TestSqueeze(TestUtils):

    @parametrize('shape', [(1, 5)])
    @parametrize('dtype', ['int32'])
    def test_squeeze_cases(self, shape, dtype):
        def op_calc(x):
            return torch.ops.aten.squeeze_.default(x)

        tensor = self._generate_tensor(shape, dtype)
        std_result = op_calc(tensor)
        compile_result = torch.compile(op_calc, backend='inductor')(tensor)
        self.assertEqual(std_result, compile_result, atol=1e-5, rtol=1e-5)

    @parametrize('shape', [(1, 5)])
    @parametrize('dtype', ['int32'])
    def test_squeeze_dims_cases(self, shape, dtype):
        def op_calc(x):
            return torch.ops.aten.squeeze_.dims(x, dim=[0])

        tensor = self._generate_tensor(shape, dtype)
        std_result = op_calc(tensor)
        compile_result = torch.compile(op_calc, backend='inductor')(tensor)
        self.assertEqual(std_result, compile_result, atol=1e-5, rtol=1e-5)


instantiate_parametrized_tests(TestSqueeze)


if __name__ == "__main__":
    run_tests()
