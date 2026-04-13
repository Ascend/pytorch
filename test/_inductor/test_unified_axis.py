import unittest
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestUnifiedAxis(TestUtils):
    """
    Test for unified-axis scenario where two input tensors have different 
    expansion patterns mapping to the same output dimension.
    
    Input 0: (256, 80, 32, 20) -> view -> (256, 80, 640)  # x = x3*20 + x4
    Input 1: (256, 80, 80, 8)  -> view -> (256, 80, 640)  # x = x5*8 + x6
    Output:  (256, 80, 640)
    """
    
    def op_calc(self, permute, permute_1):
        """
        Compute the result: view and add two tensors with different expansion patterns.
        
        Args:
            permute: Input tensor with shape (256, 80, 32, 20)
            permute_1: Input tensor with shape (256, 80, 80, 8)
        
        Returns:
            Result tensor with shape (256, 80, 640)
        """
        view_1 = permute.reshape(256, 80, 640)
        view_3 = permute_1.reshape(256, 80, 640)
        add = view_1 + view_3
        return add

    @parametrize('dtype', ['bfloat16', 'float16', 'float32'])
    def test_unified_axis_cases(self, dtype):
        """
        Test unified-axis scenario with different dtypes.
        
        This test verifies that the Inductor NPU codegen correctly handles
        the case where two input tensors have different expansion patterns
        mapping to the same output dimension.
        """
        # Generate input tensors with specific strides
        # permute: (256, 80, 32, 20) with strides (51200, 20, 1600, 1)
        permute = self._generate_tensor((256, 80, 32, 20), dtype)
        # Force specific strides by creating a contiguous tensor and then permuting
        permute = permute.as_strided((256, 80, 32, 20), (51200, 20, 1600, 1))
        
        # permute_1: (256, 80, 80, 8) with strides (51200, 8, 640, 1)
        permute_1 = self._generate_tensor((256, 80, 80, 8), dtype)
        permute_1 = permute_1.as_strided((256, 80, 80, 8), (51200, 8, 640, 1))
        
        # Compute standard result
        std_result = self.op_calc(permute, permute_1)
        
        # Compile and compute inductor result
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(permute, permute_1)
        
        # Verify results match
        self.assertEqual(std_result, inductor_result, atol=1e-2, rtol=1e-2)

    @unittest.skip("Large shape test - takes too long")
    @parametrize('dtype', ['bfloat16'])
    def test_unified_axis_large_shape(self, dtype):
        """
        Test unified-axis scenario with larger shapes.
        
        This is a stress test for larger tensor shapes.
        """
        # Larger shapes for stress testing
        permute = self._generate_tensor((512, 160, 64, 40), dtype)
        permute = permute.as_strided((512, 160, 64, 40), (204800, 40, 3200, 1))
        
        permute_1 = self._generate_tensor((512, 160, 160, 16), dtype)
        permute_1 = permute_1.as_strided((512, 160, 160, 16), (204800, 16, 1280, 1))
        
        std_result = self.op_calc(permute, permute_1)
        
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(permute, permute_1)
        
        self.assertEqual(std_result, inductor_result, atol=1e-2, rtol=1e-2)


instantiate_parametrized_tests(TestUnifiedAxis)

if __name__ == "__main__":
    run_tests()
