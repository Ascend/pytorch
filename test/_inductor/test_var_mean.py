import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
import torch_npu._inductor.config as npu_config


class TestVarMean(TestUtils):
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

    def test_welford_simt_enabled(self):
        previous = npu_config.enable_welford
        npu_config.enable_welford = True
        torch._dynamo.reset()
        try:
            input_element = self._generate_tensor((8, 64, 128), "float32")
            std_var, std_mean = self.op_calc(input_element, -1)

            compiled_op_calc = torch.compile(
                self.op_calc,
                backend="inductor",
                dynamic=False,
                options={"unroll_reductions_threshold": 1},
            )
            inductor_var, inductor_mean = compiled_op_calc(input_element, -1)

            self.assertEqual(std_var, inductor_var, atol=1e-1, rtol=1e-1, equal_nan=True)
            self.assertEqual(std_mean, inductor_mean, atol=1e-1, rtol=1e-1, equal_nan=True)
        finally:
            npu_config.enable_welford = previous
            torch._dynamo.reset()


instantiate_parametrized_tests(TestVarMean)

if __name__ == "__main__":
    run_tests()
