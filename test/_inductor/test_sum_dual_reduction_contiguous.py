import unittest
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestSumDualReductionContiguous(TestUtils):
    """Test suite for dual reduction contiguous scenario with torch.compile"""
    
    def _generate_inputs(self, dtype):
        """
        Generate random test inputs based on dtype.
        
        Args:
            dtype: Data type string ('float16', 'float32', 'bfloat16')
        
        Returns:
            tuple: (sub_38, convert_element_type_37, div_30)
        """
        dtype_map = {
            'float16': torch.float16,
            'float32': torch.float32,
            'bfloat16': torch.bfloat16
        }
        
        torch_dtype = dtype_map[dtype]
        
        sub_38 = torch.randn(256, 16, 1280, dtype=torch_dtype, device='npu')
        convert_element_type_37 = torch.randn(640, dtype=torch_dtype, device='npu')
        div_30 = torch.randn(256, 16, 1, dtype=torch_dtype, device='npu')
        
        return sub_38, convert_element_type_37, div_30
    
    def forward(self, sub_38, convert_element_type_37, div_30):
        """
        Execute the forward computation logic from sum.py.
        
        Args:
            sub_38: Input tensor of shape (256, 16, 1280)
            convert_element_type_37: Weight tensor of shape (640,)
            div_30: Divisor tensor of shape (256, 16, 1)
        
        Returns:
            torch.Tensor: Output tensor after sum operation
        """
        mul_150 = torch.ops.aten.mul.Tensor(div_30, sub_38)
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(mul_150, torch.bfloat16)
        view_94 = torch.ops.aten.view.default(convert_element_type_34, [256, 16, 32, 40])
        permute_51 = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3])
        clone_2 = torch.ops.aten.clone.default(permute_51, memory_format=torch.contiguous_format)
        view_95 = torch.ops.aten.view.default(clone_2, [256, 32, 640])
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(view_95, torch.float32)
        mul_153 = torch.ops.aten.mul.Tensor(convert_element_type_35, convert_element_type_37)
        sum_118 = torch.ops.aten.sum.dim_IntList(mul_153, [2], True)
        
        return sum_118
    
    @parametrize('dtype', ['float16', 'float32', 'bfloat16'])
    def test_sum_operation(self, dtype):
        """
        Test sum operation with different data types using torch.compile.
        
        Args:
            dtype: Data type string for parameterized testing
        """
        sub_38, convert_element_type_37, div_30 = self._generate_inputs(dtype)
        
        std_result = self.forward(sub_38, convert_element_type_37, div_30)
        
        compiled_forward = torch.compile(self.forward, backend="inductor")
        inductor_result = compiled_forward(sub_38, convert_element_type_37, div_30)
        
        torch.testing.assert_close(
            std_result,
            inductor_result,
            atol=1.0,
            rtol=0
        )

instantiate_parametrized_tests(TestSumDualReductionContiguous)

if __name__ == "__main__":
    run_tests()
