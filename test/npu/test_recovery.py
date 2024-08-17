import torch
import torch_npu

from torch_npu.npu._recovery import check_npu_tensor_is_safe, mark_all_npu_tensor_unsafe, set_npu_tensor_unsafe_check_flag
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpu(TestCase):

    def test_catch_data_check_error(self):
        torch.npu.set_device(0)
        tensor_a = torch.randn(2, 255, 255, device="npu:0")
        tensor_b = torch.randn(2, 255, 255, device="npu:0")
        set_npu_tensor_unsafe_check_flag(True)
        mark_all_npu_tensor_unsafe(0)
        self.assertFalse(check_npu_tensor_is_safe(tensor_a))
        self.assertFalse(check_npu_tensor_is_safe(tensor_b))
        with self.assertRaisesRegex(RuntimeError, "There is unsafe data in the input tensor."):
            tensor_c = tensor_a + tensor_b

    def test_mark_all_npu_tensors_unsafe_and_update_safe(self):
        torch.npu.set_device(0)
        tensor_a = torch.randn(2, 255, 255, device="npu:0")
        tensor_b = torch.randn(2, 255, 255, device="npu:0")
        self.assertTrue(check_npu_tensor_is_safe(tensor_a))
        self.assertTrue(check_npu_tensor_is_safe(tensor_b))
        # data on device 0 is marked unsafe
        mark_all_npu_tensor_unsafe(0)
        self.assertFalse(check_npu_tensor_is_safe(tensor_a))
        self.assertFalse(check_npu_tensor_is_safe(tensor_b))
        # release tensor_a and empty again, the data is safe
        del tensor_a
        tensor_a_new = torch.randn(2, 255, 255, device="npu:0")
        self.assertTrue(check_npu_tensor_is_safe(tensor_a_new))
        # d2d copy can update the unsafe tag to safe
        tensor_b.copy_(tensor_a_new)
        self.assertTrue(check_npu_tensor_is_safe(tensor_b))


if __name__ == '__main__':
    run_tests()
