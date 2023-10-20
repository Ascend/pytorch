import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestEmbeddingRenorm(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.LongTensor(np.random.uniform(0, shape[0], int(shape[0] / 2,)).astype(np.int32))

        return npu_input1, npu_input2

    def cpu_op_exec(self, input1, input2, max_norm, norm_type):
        stype = input1.dtype
        if stype == torch.float16:
            input1 = input1.float()
        output = torch.embedding_renorm_(input1, input2, max_norm=max_norm, norm_type=norm_type)
        if stype == torch.float16:
            output = output.half()
        return output

    def npu_op_exec(self, input1, input2, max_norm, norm_type):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.embedding_renorm_(input1, input2, max_norm=max_norm, norm_type=norm_type)
        output = output.to("cpu")
        return output

    def test_embedding_renorm_float16_2(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (5, 3), np.float16)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_input2 = copy.deepcopy(npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 0.1, 2)
        cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, 0.1, 2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_embedding_renorm_float16_0(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (10, 4), np.float16)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_input2 = copy.deepcopy(npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 0.2, 0)
        cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, 0.2, 0)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_embedding_renorm_float16_1(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (3, 3), np.float16)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_input2 = copy.deepcopy(npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 0.5, 1)
        cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, 0.5, 1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_embedding_renorm_float16_10(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4, 6), np.float16)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_input2 = copy.deepcopy(npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 1.0, 10)
        cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, 1.0, 10)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_embedding_renorm_float32_2(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (5, 3), np.float32)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_input2 = copy.deepcopy(npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 0.1, 2)
        cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, 0.1, 2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_embedding_renorm_float32_0(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (10, 4), np.float32)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_input2 = copy.deepcopy(npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 0.2, 0)
        cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, 0.2, 0)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_embedding_renorm_float32_1(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (3, 3), np.float32)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_input2 = copy.deepcopy(npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 0.5, 1)
        cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, 0.5, 1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_embedding_renorm_float32_10(self):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4, 6), np.float32)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_input2 = copy.deepcopy(npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, 1.0, 10)
        cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, 1.0, 10)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
