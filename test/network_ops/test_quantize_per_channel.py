import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestQuantizePerChannel(TestCase):
    def generate_data_per_channel(self, min_d, max_d, shape_x, shape_scale, shape_zp, dtype_x, dtype_scale, dtype_zp):
        input_x = np.random.uniform(min_d, max_d, shape_x).astype(dtype_x)
        scales = np.random.uniform(min_d, max_d, shape_scale).astype(dtype_scale)
        zero_points = np.random.uniform(min_d, max_d, shape_zp).astype(dtype_zp)
        npu_input_x = torch.from_numpy(input_x)
        npu_input_scales = torch.from_numpy(scales)
        npu_input_zero_points = torch.from_numpy(zero_points)
        return npu_input_x, npu_input_scales, npu_input_zero_points

    def cpu_op_exec_per_channel(self, input_x, input_scales, input_zero_points, axis, dtype, dequantize=False):
        if dequantize:
            output = torch.quantize_per_channel(input_x, input_scales, input_zero_points, axis, dtype).dequantize()
        else:
            output = torch.quantize_per_channel(input_x, input_scales, input_zero_points, axis, dtype).int_repr()
        output = output.numpy()
        return output

    def npu_op_exec_per_channel(self, input_x, input_scales, input_zero_points, axis, dtype, dequantize=False):
        input_x = input_x.to("npu")
        input_scales = input_scales.to("npu")
        input_zero_points = input_zero_points.to("npu")
        if dequantize:
            output = torch.quantize_per_channel(input_x, input_scales, input_zero_points, axis, dtype).dequantize()
        else:
            output = torch.quantize_per_channel(input_x, input_scales, input_zero_points, axis, dtype).int_repr()
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_per_channel_3_3_0_qint32(self, device="npu"):
        input_x1, scales, zero_points = self.generate_data_per_channel(-1, 1, (3, 3), (3,), (3,), np.float32,
                                                                       np.float32, np.int32)
        cpu_output1 = self.cpu_op_exec_per_channel(input_x1, scales, zero_points, 0, torch.qint32)
        npu_output1 = self.npu_op_exec_per_channel(input_x1, scales, zero_points, 0, torch.qint32)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_per_channel_3_3_3_3_1_qint8(self, device="npu"):
        input_x1, scales, zero_points = self.generate_data_per_channel(-1, 1, (3, 3), (3,), (3,), np.float32,
                                                                       np.float32, np.int8)
        cpu_output1 = self.cpu_op_exec_per_channel(input_x1, scales, zero_points, 1, torch.qint8)
        npu_output1 = self.npu_op_exec_per_channel(input_x1, scales, zero_points, 1, torch.qint8)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_per_channel_3_3_3_3_3_3_3_3_4_quint8(self, device="npu"):
        input_x1, scales, zero_points = self.generate_data_per_channel(-1, 1, (3, 3, 3, 3, 3, 3, 3, 3), (3,), (3,),
                                                                       np.float32, np.float32, np.int32)
        cpu_output1 = self.cpu_op_exec_per_channel(input_x1, scales, zero_points, 4, torch.quint8)
        npu_output1 = self.npu_op_exec_per_channel(input_x1, scales, zero_points, 4, torch.quint8)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_per_channel_30_30_30_30_30_2_quint8(self, device="npu"):
        input_x1, scales, zero_points = self.generate_data_per_channel(-1, 1, (30, 30, 30, 30), (30,), (30,),
                                                                       np.float32, np.float32, np.uint8)
        cpu_output1 = self.cpu_op_exec_per_channel(input_x1, scales, zero_points, 2, torch.quint8)
        npu_output1 = self.npu_op_exec_per_channel(input_x1, scales, zero_points, 2, torch.quint8)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_per_channel_dequantize_30_30_30_30_30_2_quint8(self, device="npu"):
        input_x1, scales, zero_points = self.generate_data_per_channel(-1, 1, (30, 30, 30, 30), (30,), (30,),
                                                                       np.float32, np.float32, np.uint8)
        cpu_output1 = self.cpu_op_exec_per_channel(input_x1, scales, zero_points, 2, torch.quint8, dequantize=True)
        npu_output1 = self.npu_op_exec_per_channel(input_x1, scales, zero_points, 2, torch.quint8, dequantize=True)
        self.assertRtolEqual(cpu_output1, npu_output1)


if __name__ == "__main__":
    run_tests()
