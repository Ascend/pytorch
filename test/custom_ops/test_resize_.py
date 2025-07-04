import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestResize(TestCase):

    def test_masked_select_out(self):

        input_data = torch.tensor([[[[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]]]]], dtype=torch.float)
        mask = torch.tensor([True, False, True, False, True])

        input_data_npu = input_data.npu()
        mask_npu = mask.npu()

        out_tensor = torch.empty((1, 1, 1, 1, 1), dtype=input_data.dtype)
        out_tensor_npu = out_tensor.npu()

        out_tensor_npu = out_tensor_npu.view(-1)
        out_tensor_npu = torch.masked_select(input_data_npu, mask_npu, out=out_tensor_npu)
        out_tensor = torch.masked_select(input_data, mask, out=out_tensor)
        self.assertRtolEqual(out_tensor_npu, out_tensor)
    
    def test_resize_ncdhw(self):
        out_tensor = torch.empty((1, 1, 1, 1, 1), dtype=torch.float16).npu()
        shape = [25]
        out_tensor.resize_(shape)
        self.assertEqual(shape, out_tensor.shape)

if __name__ == "__main__":
    run_tests()
