import os

from unittest import skip

import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestFastAutotuneAbs(TestUtils):
    def op_calc(self, first_element):
        result = torch.abs(first_element)
        return result

    def setUp(self):
        self.original_fastautotune = os.environ.get("FASTAUTOTUNE")
        self.original_compile_threads = torch._inductor.config.compile_threads
        os.environ["FASTAUTOTUNE"] = "1"
        import torch_npu
        import torch_npu._inductor

    def tearDown(self):
        torch._inductor.config.compile_threads = self.original_compile_threads
        if self.original_fastautotune is not None:
            os.environ["FASTAUTOTUNE"] = self.original_fastautotune
        else:
            del os.environ["FASTAUTOTUNE"]
        import torch_npu
        import torch_npu._inductor

    @skip("skip ci codegen error")
    @parametrize('shape', [(1024, 32), (256, 8)])
    @parametrize('dtype', ['float16', 'float32', 'bfloat16'])
    def test_pointwise_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)

instantiate_parametrized_tests(TestFastAutotuneAbs)

if __name__ == "__main__":
    run_tests()