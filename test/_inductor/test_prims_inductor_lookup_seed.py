import torch
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils
import unittest


class TestPrimsinductor_lookup_seed(TestUtils):
    @unittest.skip("not supported yet")
    @parametrize("shape", [(128,)])
    @parametrize("dtype", ["int8", "int32", "int64", "bool", "float16", "bfloat16", "float32"])
    def test_prims_inductor_lookup_seed_default(self, shape, dtype):
        def op_calc(seeds, index):
            out = torch.ops.prims.inductor_lookup_seed.default(seeds, index)
            return out

        seeds = self._generate_tensor(shape, dtype)
        index = 0
        std_result = op_calc(seeds, index)
        compiled_op_calc = torch.compile(op_calc, backend="inductor")
        inductor_result = compiled_op_calc(seeds, index)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestPrimsinductor_lookup_seed)

if __name__ == "__main__":
    run_tests()