import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestMatMul(TestCase):
    def op_exec_cpu(self, mat1, mat2):
        input1 = mat1
        input2 = mat2
        input1.requires_grad = True
        input2.requires_grad = True

        cpu_output = torch.matmul(input1, input2)
        tmp = torch.ones_like(cpu_output)
        cpu_output.backward(tmp)

        return cpu_output.detach().numpy(), input1.grad.numpy(), input2.grad.numpy()

    def op_exec_npu(self, mat1, mat2):
        input1 = mat1
        input2 = mat2
        input1.requires_grad = True
        input2.requires_grad = True

        npu_output = torch.matmul(input1, input2)
        tmp = torch.ones_like(npu_output)
        npu_output.backward(tmp)
        npu_output = npu_output.cpu()
        return npu_output.detach().cpu().numpy(), input1.grad.cpu().numpy(), input2.grad.cpu().numpy()

    def matmul_backward_result(self, shape_format):
        for item in shape_format:
            mat1_cpu, mat1_npu = create_common_tensor(item[0], -10, 10)
            if mat1_cpu.dtype == torch.float16:
                mat1_cpu = mat1_cpu.to(torch.float32)
            mat2_cpu, mat2_npu = create_common_tensor(item[1], -10, 10)
            if mat2_cpu.dtype == torch.float16:
                mat2_cpu = mat2_cpu.to(torch.float32)
            cpu_output, cpu_mat1_grad, cpu_mat2_grad = self.op_exec_cpu(mat1_cpu, mat2_cpu)
            npu_output, npu_mat1_grad, npu_mat2_grad = self.op_exec_npu(mat1_npu, mat2_npu)

            self.assertRtolEqual(cpu_output.astype(npu_output.dtype), npu_output)
            self.assertRtolEqual(cpu_mat1_grad.astype(npu_mat1_grad.dtype), npu_mat1_grad)
            self.assertRtolEqual(cpu_mat2_grad.astype(npu_mat2_grad.dtype), npu_mat2_grad)

    @unittest.skip("skip test_matmul_backward_shape_format_fp16_case1 now")
    def test_matmul_backward_shape_format_fp16_case1(self, device="npu"):
        shape_format = [
            # mat1 1dim, mat2 1dim
            [[np.float16, 2, [5]], [np.float16, 2, [5]]],
            [[np.float16, 2, [16]], [np.float16, 2, [16]]],
        ]
        self.matmul_backward_result(shape_format)

    @unittest.skip("skip test_matmul_backward_shape_format_fp16_case3 now")
    def test_matmul_backward_shape_format_fp16_case3(self, device="npu"):
        shape_format = [
            # mat1 1dim, mat2 2dim, mat1 2dim, mat2 1dim
            [[np.float16, 2, [5]], [np.float16, 2, [5, 6]]],
            [[np.float16, 2, [5]], [np.float16, 2, [5, 5]]],
            [[np.float16, 2, [3, 4]], [np.float16, 2, [4]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case4(self, device="npu"):
        shape_format = [
            # mat1 2dim, mat2 2dim
            [[np.float16, 2, [5, 7]], [np.float16, 2, [7, 10]]],
            [[np.float16, 2, [5, 10]], [np.float16, 2, [10, 20]]],
        ]
        self.matmul_backward_result(shape_format)

    @unittest.skip("skip test_matmul_backward_shape_format_fp16_case5 now")
    def test_matmul_backward_shape_format_fp16_case5(self, device="npu"):
        shape_format = [
            # mat1 >2dim, mat2 1dim
            [[np.float16, 2, [4, 5, 10]], [np.float16, 2, [10]]],
            [[np.float16, 2, [5, 10, 20, 30]], [np.float16, 2, [30]]],
            [[np.float16, 2, [20, 30, 40, 50, 60]], [np.float16, 2, [60]]],
            [[np.float16, 2, [2, 3, 4, 5, 6, 8]], [np.float16, 2, [8]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case6(self, device="npu"):
        shape_format = [
            # mat1 >2dim, mat2 2dim
            [[np.float16, 2, [5, 7, 10]], [np.float16, 2, [10, 16]]],
            [[np.float16, 2, [5, 10, 20, 30]], [np.float16, 2, [30, 25]]],
            [[np.float16, 2, [2, 5, 7, 8, 9, 10]], [np.float16, 2, [10, 16]]],
        ]
        self.matmul_backward_result(shape_format)

    @unittest.skip("skip test_matmul_backward_shape_format_fp16_case7 now")
    def test_matmul_backward_shape_format_fp16_case7(self, device="npu"):
        shape_format = [
            # mat1 1dim, mat2 >2dim
            [[np.float16, 2, [3, ]], [np.float16, 2, [2, 3, 2]]],
            [[np.float16, 2, [20]], [np.float16, 2, [5, 10, 20, 30]]]
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case8(self, device="npu"):
        shape_format = [
            # mat1 2dim, mat2 >2dim
            [[np.float16, 2, [2, 3]], [np.float16, 2, [2, 3, 2]]],
            [[np.float16, 2, [44, 20]], [np.float16, 2, [5, 10, 20, 30]]],
            [[np.float16, 2, [75, 50]], [np.float16, 2, [2, 3, 40, 50, 60]]],
        ]
        self.matmul_backward_result(shape_format)

    def test_matmul_backward_shape_format_fp16_case9(self, device="npu"):
        shape_format = [
            [[np.float16, 2, [5, 7, 10]], [np.float16, 2, [5, 10, 15]]],
            [[np.float16, 2, [68, 75, 16]], [np.float16, 2, [68, 16, 43]]],
            # TODO(ascend): Insufficient precision
            # 在两个输入shape不一致的情况下,会通过expand将两个tensor shape对齐。反向时expand的反向会调用sum(dim)，在fp16下与CPU比较不过。
            # 但是结果与CUDA比对通过。所以只放开两个tensor batch部分一致的用例
        ]
        self.matmul_backward_result(shape_format)

    @unittest.skip("skip test_matmul_allow_hf32 now")
    def test_matmul_allow_hf32(self, device="npu"):
        torch.npu.matmul.allow_hf32 = True
        shape_format = [
            # mat1 1dim, mat2 1dim
            [[np.float16, 2, [5]], [np.float16, 2, [5]]],
            [[np.float16, 2, [16]], [np.float16, 2, [16]]],
        ]
        self.matmul_backward_result(shape_format)
        torch.npu.matmul.allow_hf32 = False


if __name__ == "__main__":
    run_tests()
