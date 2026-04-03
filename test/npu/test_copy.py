import unittest
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestCopyKernelMemoryFormat(TestCase):
    def test_h2d_copy_contiguous_tensor(self):
        dtype_list = [np.float16, np.float32, np.int32, np.int64]
        shape_list = [[10, 20], [32, 64, 128], [2, 3, 4, 5]]
        shape_format = [
            [dtype, 2, shape] for dtype in dtype_list for shape in shape_list
        ]
        
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -100, 100)
            npu_input_copy = cpu_input.npu()
            self.assertRtolEqual(npu_input_copy.cpu().numpy(), cpu_input.numpy())

    def test_h2d_copy_non_contiguous_tensor(self):
        dtype_list = [np.float16, np.float32]
        shape_list = [[32, 64], [16, 32, 64]]
        shape_format = [
            [dtype, 2, shape] for dtype in dtype_list for shape in shape_list
        ]
        
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -100, 100)
            cpu_transposed = cpu_input.transpose(-1, -2)
            npu_transposed = cpu_transposed.npu()
            npu_contiguous = npu_transposed.contiguous()
            self.assertRtolEqual(npu_contiguous.cpu().numpy(), cpu_transposed.contiguous().numpy())

    def test_d2h_copy_contiguous_tensor(self):
        dtype_list = [np.float16, np.float32, np.int32, np.int64]
        shape_list = [[10, 20], [32, 64, 128], [2, 3, 4, 5]]
        shape_format = [
            [dtype, 2, shape] for dtype in dtype_list for shape in shape_list
        ]
        
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -100, 100)
            cpu_output = npu_input.cpu()
            self.assertRtolEqual(cpu_output.numpy(), cpu_input.numpy())

    def test_d2h_copy_non_contiguous_tensor(self):
        dtype_list = [np.float16, np.float32]
        shape_list = [[32, 64], [16, 32, 64]]
        shape_format = [
            [dtype, 2, shape] for dtype in dtype_list for shape in shape_list
        ]
        
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -100, 100)
            npu_transposed = npu_input.transpose(-1, -2)
            cpu_output = npu_transposed.cpu()
            self.assertRtolEqual(cpu_output.numpy(), cpu_input.transpose(-1, -2).contiguous().numpy())

    def test_h2d_copy_different_dtype(self):
        src_dtype_list = [np.float32, np.float16]
        dst_dtype_list = [torch.float16, torch.float32]
        shape = [32, 64]
        
        for src_dtype, dst_dtype in zip(src_dtype_list, dst_dtype_list):
            cpu_input = torch.randn(shape, dtype=torch.float32) * 100
            cpu_input = cpu_input.to(torch.from_numpy(np.array([])).dtype if src_dtype == np.float32 else torch.float16)
            
            npu_input = cpu_input.npu()
            npu_output = npu_input.to(dst_dtype)
            
            cpu_output = cpu_input.to(dst_dtype)
            self.assertRtolEqual(npu_output.cpu().numpy(), cpu_output.numpy())

    def test_d2h_copy_different_dtype(self):
        dtype_pairs = [
            (np.float16, torch.float32),
            (np.float32, torch.float16),
        ]
        shape = [32, 64]
        
        for src_dtype, dst_dtype in dtype_pairs:
            cpu_input, npu_input = create_common_tensor([src_dtype, 0, shape], -100, 100)
            cpu_output = npu_input.cpu().to(dst_dtype)
            
            expected = cpu_input.to(dst_dtype)
            self.assertRtolEqual(cpu_output.numpy(), expected.numpy())

    def test_h2d_copy_slice_tensor(self):
        shape = [64, 128]
        cpu_input = torch.randn(shape)
        
        cpu_slice = cpu_input[10:30, 20:60]
        npu_slice = cpu_slice.npu()
        
        npu_contiguous = npu_slice.contiguous()
        self.assertRtolEqual(npu_contiguous.cpu().numpy(), cpu_slice.contiguous().numpy())

    def test_d2h_copy_slice_tensor(self):
        shape = [64, 128]
        cpu_input = torch.randn(shape)
        npu_input = cpu_input.npu()
        
        npu_slice = npu_input[10:30, 20:60]
        cpu_slice = npu_slice.cpu()
        self.assertRtolEqual(cpu_slice.numpy(), cpu_input[10:30, 20:60].contiguous().numpy())

    def test_d2h_copy_broadcast_tensor(self):
        shape = [1, 64, 1]
        cpu_input = torch.randn(shape)
        npu_input = cpu_input.npu()
        
        npu_broadcast = npu_input.expand(4, 64, 128)
        cpu_output = npu_broadcast.cpu()
        self.assertRtolEqual(cpu_output.numpy(), cpu_input.expand(4, 64, 128).contiguous().numpy())

    def test_h2d_copy_permute_tensor(self):
        shape = [32, 64, 128]
        cpu_input = torch.randn(shape)
        cpu_permuted = cpu_input.permute(2, 0, 1)
        npu_permuted = cpu_permuted.npu()
        npu_contiguous = npu_permuted.contiguous()
        self.assertRtolEqual(npu_contiguous.cpu().numpy(), cpu_permuted.contiguous().numpy())

    def test_d2h_copy_permute_tensor(self):
        shape = [32, 64, 128]
        cpu_input = torch.randn(shape)
        npu_input = cpu_input.npu()
        
        npu_permuted = npu_input.permute(2, 0, 1)    
        cpu_output = npu_permuted.cpu()
        self.assertRtolEqual(cpu_output.numpy(), cpu_input.permute(2, 0, 1).contiguous().numpy())


if __name__ == "__main__":
    run_tests()