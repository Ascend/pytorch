import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
from torch._inductor.utils import run_and_get_code
import torch_npu
import torch_npu._inductor


class TestAdd(TestUtils):
    

    def op_calc(self, first_element, second_element):
        result = first_element + second_element
        return result

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float32', 'int64'])
    def test_options_environ_cases(self, shape, dtype):
        x = self._generate_tensor(shape, dtype)
        y = self._generate_tensor(shape, dtype)
        std_out = self.op_calc(x, y)
        compile_func = torch.compile(self.op_calc, options={"npu_backend": "mlir"})
        compile_out, codes = run_and_get_code(compile_func, x, y)
        self.assertEqual(std_out, compile_out)
        self.assertTrue("mlir_fused" in codes[0])

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float32', 'int64'])
    def test_config_environ_cases(self, shape, dtype):
        torch._inductor.config.npu_backend = "mlir"
        x = self._generate_tensor(shape, dtype)
        y = self._generate_tensor(shape, dtype)
        std_out = self.op_calc(x, y)
        compile_func = torch.compile(self.op_calc)
        compile_out, codes = run_and_get_code(compile_func, x, y)
        self.assertEqual(std_out, compile_out)
        self.assertTrue("mlir_fused" in codes[0])


instantiate_parametrized_tests(TestAdd)

if __name__ == "__main__":
    run_tests()