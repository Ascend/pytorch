import os

import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor


class TestFastAutotuneVarMean(TestUtils):
    def setUp(self):
        self.original_fastautotune = os.environ.get("FASTAUTOTUNE")
        self.original_compile_threads = torch._inductor.config.compile_threads
        os.environ["FASTAUTOTUNE"] = "1"

    def tearDown(self):
        torch._inductor.config.compile_threads = self.original_compile_threads
        if self.original_fastautotune is not None:
            os.environ["FASTAUTOTUNE"] = self.original_fastautotune
        else:
            del os.environ["FASTAUTOTUNE"]

    def op_calc(self, input_element, dim):
        return torch.var_mean(input_element, dim)

    # case：The shape must not be too large
    @parametrize('shape', [(8, 64, 128)])
    @parametrize('dim', [0, 1, 2, (0, 2), (0, 1)])
    @parametrize('dtype', ['float32'])
    def test_reduction_cases_shapes(self, shape, dim, dtype):
        input_element = self._generate_tensor(shape, dtype)
        std_var, std_mean = self.op_calc(input_element, dim)
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor", dynamic=False)
        inductor_var, inductor_mean = compiled_op_calc(input_element, dim)
        self.assertEqual(std_var, inductor_var, atol=1e-1, rtol=1e-1, equal_nan=True)
        self.assertEqual(std_mean, inductor_mean, atol=1e-1, rtol=1e-1, equal_nan=True)


instantiate_parametrized_tests(TestFastAutotuneVarMean)

if __name__ == "__main__":
    run_tests()
