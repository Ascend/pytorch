"""
Test cases for NPUSwappedMemoryAllocator and svm_deleter functionality
including async stream synchronization scenarios

To run this test:
1. Build torch_npu: bash ci/build.sh --python=3.10
2. Install: pip install dist/torch_npu-*.whl
3. Run: python test/npu/test_swapped_memory_allocator.py
"""

import unittest
import sys
import gc
import weakref
import threading

import torch
import torch_npu
from torch_npu.testing.common_utils import SupportedDevices



class TestNPUSwappedMemoryAllocator(unittest.TestCase):

    def setUp(self):
        super().setUp()
        torch.npu.empty_cache()
        gc.collect()

    def tearDown(self):
        gc.collect()
        torch.npu.empty_cache()
        gc.collect()
        super().tearDown()

    @SupportedDevices(['Ascend910B'])
    def test_01_basic_allocation_and_deallocation(self):
        """Test 1: Basic allocation and deallocation with svm_deleter"""
        for _ in range(3):
            tensors = []
            for i in range(5):
                tensor = torch_npu.empty_with_swapped_memory(
                    [256 * (i + 1)], 
                    dtype=torch.float32, 
                    device='npu:0'
                )
                tensor.fill_(float(i))
                tensors.append(tensor)
            
            for t in tensors:
                del t
            del tensors
            gc.collect()
            torch.npu.empty_cache()
            gc.collect()


    @SupportedDevices(['Ascend910B'])
    def test_02_async_operations_and_release(self):
        """Test 2: Async operations followed by release (verify stream sync works)"""
        tensor = torch_npu.empty_with_swapped_memory(
            [512, 512], 
            dtype=torch.float32, 
            device='npu:0'
        )
        
        tensor.fill_(1.0)
        
        for i in range(10):
            tensor = tensor * 1.1
            tensor = tensor + i * 0.1
        
        tensor.sqrt_()
        
        weak_ref = weakref.ref(tensor)
        del tensor
        gc.collect()
        
        self.assertIsNone(weak_ref())
        
        torch.npu.empty_cache()


if __name__ == '__main__':
    unittest.main(verbosity=2)
