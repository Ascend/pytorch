import torch
from torch.utils.dlpack import to_dlpack, from_dlpack
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.decorator import Dtypes, instantiate_tests


@instantiate_tests
class TestDLPack(TestCase):

    @Dtypes(torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)
    def test_dlpack_roundtrip_basic(self, dtype, device="npu"):
        """Test basic dlpack roundtrip: torch_npu tensor -> dlpack -> torch_npu tensor"""
        # Create original tensor
        if dtype == torch.bool:
            original = torch.randint(0, 2, (2, 3, 4), dtype=dtype, device=device)
        elif dtype == torch.uint8:
            original = torch.randint(0, 10, (2, 3, 4), dtype=dtype, device=device)
        elif dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            original = torch.randint(-10, 10, (2, 3, 4), dtype=dtype, device=device)
        else:
            original = torch.randn(2, 3, 4, dtype=dtype, device=device)
        
        # Convert to dlpack
        dlpack_tensor = to_dlpack(original)
        
        # Convert back to torch_npu tensor
        restored = from_dlpack(dlpack_tensor)
        
        # Verify the roundtrip
        self.assertEqual(original, restored)
        self.assertEqual(original.dtype, restored.dtype)
        self.assertEqual(original.device, restored.device)
        self.assertEqual(original.shape, restored.shape)
        self.assertEqual(original.stride(), restored.stride())

    @Dtypes(torch.half, torch.float, torch.double)
    def test_dlpack_roundtrip_different_shapes(self, dtype, device="npu"):
        """Test dlpack roundtrip with different tensor shapes"""
        shapes = [
            (1,),           # 1D tensor
            (5, 5),         # 2D tensor
            (2, 3, 4),      # 3D tensor
            (2, 2, 2, 2),   # 4D tensor
            (1, 1, 1, 1, 1) # 5D tensor
        ]
        
        for shape in shapes:
            with self.subTest(shape=shape):
                original = torch.randn(shape, dtype=dtype, device=device)
                dlpack_tensor = to_dlpack(original)
                restored = from_dlpack(dlpack_tensor)
                
                self.assertEqual(original, restored)
                self.assertEqual(original.shape, restored.shape)

    @Dtypes(torch.float)
    def test_dlpack_roundtrip_contiguous(self, dtype, device="npu"):
        """Test dlpack roundtrip with contiguous and non-contiguous tensors"""
        # Test contiguous tensor
        original_contiguous = torch.randn(4, 4, dtype=dtype, device=device)
        self.assertTrue(original_contiguous.is_contiguous())
        
        dlpack_tensor = to_dlpack(original_contiguous)
        restored = from_dlpack(dlpack_tensor)
        
        self.assertEqual(original_contiguous, restored)
        self.assertTrue(restored.is_contiguous())
        
        # Test non-contiguous tensor (transpose)
        original_non_contiguous = original_contiguous.t()
        self.assertFalse(original_non_contiguous.is_contiguous())
        
        dlpack_tensor = to_dlpack(original_non_contiguous)
        restored = from_dlpack(dlpack_tensor)
        
        self.assertEqual(original_non_contiguous, restored)
        self.assertEqual(original_non_contiguous.stride(), restored.stride())

    @Dtypes(torch.float)
    def test_dlpack_memory_sharing(self, dtype, device="npu"):
        """Test that dlpack shares memory with original tensor"""
        original = torch.randn(3, 3, dtype=dtype, device=device)
        original_data_ptr = original.data_ptr()
        
        # Convert to dlpack and back
        dlpack_tensor = to_dlpack(original)
        restored = from_dlpack(dlpack_tensor)
        
        # Check if memory is shared (data_ptr should be the same)
        self.assertEqual(original_data_ptr, restored.data_ptr())
        
        # Modify original tensor and check if restored tensor is also modified
        original.fill_(42.0)
        self.assertEqual(original, restored)

    @Dtypes(torch.bfloat16)
    def test_dlpack_bfloat16_support(self, dtype, device="npu"):
        """Test dlpack with bfloat16 data type"""
        original = torch.randn(3, 4, dtype=dtype, device=device)
        dlpack_tensor = to_dlpack(original)
        restored = from_dlpack(dlpack_tensor)
        
        self.assertEqual(original, restored)
        self.assertEqual(original.dtype, restored.dtype)
    
    @Dtypes(torch.float)
    def test_dlpack_cpu(self, dtype, device="cpu"):
        """Test that dlpack shares memory with original cpu tensor"""
        original = torch.randn(3, 3, dtype=dtype, device=device)
        original_data_ptr = original.data_ptr()
        
        # Convert to dlpack and back
        dlpack_tensor = to_dlpack(original)
        restored = from_dlpack(dlpack_tensor)
        
        # Check if memory is shared (data_ptr should be the same)
        self.assertEqual(original_data_ptr, restored.data_ptr())


if __name__ == '__main__':
    run_tests()