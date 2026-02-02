import os

from unittest import skip

import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestFastAutotuneClone(TestUtils):
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

    def op_calc(self, input_element, dim):
        return torch.clone(input_element)

    @skip("skip ci codegen error")
    @parametrize('shape', [(8, 64, 128)])
    @parametrize('dim', [0])
    @parametrize('dtype', ['float32'])
    def test_fast_autotune_duction_cases_shapes(self, shape, dim, dtype):
        input_element = self._generate_tensor(shape, dtype)
        std_ret = self.op_calc(input_element, dim)
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_ret = compiled_op_calc(input_element, dim)
        self.assertEqual(std_ret, inductor_ret, equal_nan=True)

instantiate_parametrized_tests(TestFastAutotuneClone)

if __name__ == "__main__":
    run_tests()
