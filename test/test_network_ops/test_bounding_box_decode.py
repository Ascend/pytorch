import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TesBoundingBoxDecode(TestCase):
    def test_decode_shape_format_fp32(self):
        input1 = torch.tensor([[1., 2., 3., 4.], [3., 4., 5., 6.]], dtype=torch.float32).to("npu")
        input2 = torch.tensor([[5., 6., 7., 8.], [7., 8., 9., 6.]], dtype=torch.float32).to("npu")
        input1_fp16 = input1.to(torch.half)
        input2_fp16 = input2.to(torch.half)
        expedt_output = torch.tensor([[2.5000, 6.5000, 9.0000, 9.0000],
                                      [9.0000, 9.0000, 9.0000, 9.0000]], dtype=torch.float32)
        output = torch_npu.npu_bounding_box_decode(input1, input2, 0, 0, 0, 0, 1, 1, 1, 1, (10, 10), 0.1)
        output_fp16 = torch_npu.npu_bounding_box_decode(input1_fp16, input2_fp16, 0, 0, 0, 0, 1, 1, 1, 1, (10, 10), 0.1)
        self.assertRtolEqual(expedt_output, output.cpu())
        self.assertRtolEqual(expedt_output.to(torch.half), output_fp16.cpu())


if __name__ == "__main__":
    run_tests()
