import os

from unittest import skip

import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestFastAutotuneRelu(TestUtils):
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

    def op_calc(self, first_element):
        result = torch.relu(first_element)
        return result

    @skip("skip ci codegen error")
    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float16', 'float32', 'bfloat16', 'int32', 'int64'])
    def test_pointwise_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)
        std_result = self.op_calc(first_element)
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element)
        self.assertEqual(std_result, inductor_result)

instantiate_parametrized_tests(TestFastAutotuneRelu)

if __name__ == "__main__":
    run_tests()
