import unittest
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestFactoryReduction(TestUtils):
    def op_calc(self, shape, dtype_val):
        # Use dtype_val which is actual torch dtype instead of torch.float32 directly
        # based on the parameter passed
        full_tensor = torch.ops.aten.full.default(shape, 1.0, dtype=dtype_val, layout=torch.strided, device='npu', pin_memory=False)
        result = torch.ops.aten.amax.default(full_tensor, [2])
        return result

    @parametrize('shape', [(2, 86, 8)])
    @parametrize('dtype', ['float32'])
    def test_factory_reduction(self, shape, dtype):
        # Get actual torch dtype from string
        dtype_val = getattr(torch, dtype)
        
        std_result = self.op_calc(shape, dtype_val)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(shape, dtype_val)
        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)

instantiate_parametrized_tests(TestFactoryReduction)

if __name__ == "__main__":
    run_tests()
