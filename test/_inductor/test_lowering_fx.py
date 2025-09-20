import os
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from torch._inductor.utils import run_and_get_code
from testutils import TestUtils
import torch_npu

torch._inductor.config.fx_graph_cache = False
os.environ["INDUCTOR_ASCEND_CHECK_ACCURACY"] = "1"


class TestLoweringFx(TestUtils):
    @parametrize('shape', [(32, 16, 64, 128)])
    @parametrize('dim', [0])
    @parametrize('dtype', ['float32'])
    def test_sum_not_fallback(self, shape, dim, dtype):
        input_element = self._generate_tensor(shape, dtype, floatPOSIFLAG=1)
        golden = torch.sum(input_element, dim)
        compiled_op_calc = torch.compile(torch.sum, backend="inductor", dynamic=False)
        inductor_result, output_codes = run_and_get_code(compiled_op_calc, input_element, dim)
        self.assertTrue(len(output_codes) == 1)
        self.assertTrue(output_codes[0].count("async_compile.triton") == 1)
        self.assertTrue(output_codes[0].count(".run(") == 1)
        torch.testing.assert_close(golden, inductor_result, rtol=1e-4, atol=1e-4)

    @parametrize('shape', [(32, 16, 64, 128)])
    @parametrize('dtype', ['float32'])
    def test_div_by_reciprocal_mul(self, shape, dtype):
        input_element = self._generate_tensor(shape, dtype)
        divisor = 128.0
        golden = torch.div(input_element, divisor)
        compiled_op_calc = torch.compile(torch.div, backend="inductor", dynamic=False)
        inductor_result, output_codes = run_and_get_code(compiled_op_calc, input_element, divisor)
        self.assertTrue(len(output_codes) == 1)
        self.assertTrue(output_codes[0].count("async_compile.triton") == 1)
        self.assertTrue("torch.ops.aten.div.Tensor(" not in output_codes[0])
        self.assertTrue("torch.ops.aten.mul.Tensor(" in output_codes[0])
        torch.testing.assert_close(golden, inductor_result, rtol=1e-4, atol=1e-4)


instantiate_parametrized_tests(TestLoweringFx)

if __name__ == "__main__":
    run_tests()