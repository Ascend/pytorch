import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.contrib import transfer_to_npu


class TestTransferToNpu(TestCase):

    def test_wrap_isinstance(self):
        # check builtins isinstance grammar
        self.assertTrue(isinstance(1, int))
        self.assertTrue(isinstance(1, (int, str)))
        self.assertFalse(isinstance(1, str))
        with self.assertRaises(TypeError):
            isinstance(1, [str, int])

        # check torch.device
        self.assertFalse(isinstance(1, torch.device))

        # check torch.cuda.device
        device = -1
        torch.cuda.device(device)

    def test_amp_function(self):
        self.assertEqual(torch.cuda.amp.autocast_mode, torch_npu.npu.amp.autocast_mode)
        self.assertEqual(torch.cuda.amp.common, torch_npu.npu.amp.common)
        self.assertEqual(torch.cuda.amp.grad_scaler, torch_npu.npu.amp.grad_scaler)

    def test_wrap_device(self):
        device = torch.device(f"cuda:{0}")
        torch.cuda.set_device(device)
        a = torch.randint(1, 5, (2, 3), device=device)
        self.assertEqual(a.device.type, 'npu')

    def test_wrap_device_int_type(self):
        a = torch.rand(1)
        device_id = torch.cuda.current_device()
        b = a.to(device=device_id)
        c = a.to(device_id)
        d = torch.tensor(1, device=device_id)

    def test_wrap_autocast(self):
        a = torch.randn([3, 3], device='cuda')
        b = torch.randn([3, 3], device='cuda')
        c = a @ b
        c_before_autocast = c.dtype

        with torch.autocast('cuda'):
            c = a @ b
        c_after_autocast_args = c.dtype

        with torch.autocast(device_type='cuda'):
            c = a @ b
        c_after_autocast_kwargs = c.dtype

        self.assertNotEqual(c_before_autocast, c_after_autocast_args)       
        self.assertEqual(c_after_autocast_args, c_after_autocast_kwargs)

    def test_device_meta(self):
        with torch.device('meta'):
            out = torch.nn.Linear(32, 8)
        device = out.weight.device
        self.assertEqual(device.type, 'meta')

    def test_set_default_device(self):
        torch.set_default_device("cuda")
        a = torch.tensor(1)
        self.assertEqual(a.device.type, 'npu')

    def test_device_context(self):
        device = torch.device('cuda')
        with device:
            a = torch.tensor(1)
        self.assertEqual(a.device.type, 'npu')


if __name__ == "__main__":
    run_tests()
