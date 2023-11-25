import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestQuantizePerTensor(TestCase):

    def generate_data_per_tensor(self, min_d, max_d, shape_x, dtype_x):
        input_x = np.random.uniform(min_d, max_d, shape_x).astype(dtype_x)
        npu_input_x = torch.from_numpy(input_x)
        return npu_input_x

    def cpu_op_exec_per_tensor(self, input_x, input_scale, input_zero_point, dtype, dequantize=False):
        if dequantize:
            output = torch.quantize_per_tensor(input_x, input_scale, input_zero_point, dtype).dequantize()
        else:
            output = torch.quantize_per_tensor(input_x, input_scale, input_zero_point, dtype).int_repr()
        output = output.numpy()
        return output

    def npu_op_exec_per_tensor(self, input_x, input_scale, input_zero_point, dtype, dequantize=False):
        input_x = input_x.to("npu")
        if dequantize:
            output = torch.quantize_per_tensor(input_x, input_scale, input_zero_point, dtype).dequantize()
        else:
            output = torch.quantize_per_tensor(input_x, input_scale, input_zero_point, dtype).int_repr()
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_per_tensor_3_3_0p1_10_qint32(self, device="npu"):
        input_x1 = self.generate_data_per_tensor(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec_per_tensor(input_x1, 0.1, 10, torch.qint32)
        npu_output1 = self.npu_op_exec_per_tensor(input_x1, 0.1, 10, torch.qint32)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_per_tensor_3_3_0p1_10_qint8(self, device="npu"):
        input_x1 = self.generate_data_per_tensor(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec_per_tensor(input_x1, 0.1, 10, torch.qint8)
        npu_output1 = self.npu_op_exec_per_tensor(input_x1, 0.1, 10, torch.qint8)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_per_tensor_3_3_3_3_3_3_0p1_10_quint8(self, device="npu"):
        input_x1 = self.generate_data_per_tensor(-1, 1, (3, 3, 3, 3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec_per_tensor(input_x1, 0.1, 10, torch.quint8)
        npu_output1 = self.npu_op_exec_per_tensor(input_x1, 0.1, 10, torch.quint8)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_per_tensor_30_30_30_30_30_30_0p01_5_quint8(self, device="npu"):
        input_x1 = self.generate_data_per_tensor(-1, 1, (30, 30, 30, 30, 30, 30), np.float32)
        cpu_output1 = self.cpu_op_exec_per_tensor(input_x1, 0.01, 5, torch.quint8)
        npu_output1 = self.npu_op_exec_per_tensor(input_x1, 0.01, 5, torch.quint8)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_per_tensor_dequantize_30_30_30_30_30_30_0p01_5_quint8(self, device="npu"):
        input_x1 = self.generate_data_per_tensor(-1, 1, (30, 30, 30, 30, 30, 30), np.float32)
        cpu_output1 = self.cpu_op_exec_per_tensor(input_x1, 0.01, 5, torch.quint8, dequantize=True)
        npu_output1 = self.npu_op_exec_per_tensor(input_x1, 0.01, 5, torch.quint8, dequantize=True)
        self.assertRtolEqual(cpu_output1, npu_output1)


if __name__ == "__main__":
    run_tests()
