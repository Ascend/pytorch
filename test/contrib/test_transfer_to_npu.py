from unittest.mock import patch

import torch
from torch.nn.parameter import UninitializedTensorMixin
from torch.utils.data import TensorDataset
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.contrib.transfer_to_npu import _del_nccl_device_backend_map


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

    def test_data_loader_pin_memory(self):
        images = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        labels = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
        dataset = TensorDataset(images, labels)
        data_loader = torch.utils.data.DataLoader(
            dataset, num_workers=2, pin_memory=True, pin_memory_device='cuda')

        data_iter = iter(data_loader)
        images0, _ = next(data_iter)
        self.assertTrue(images0.is_pinned())

    def test_replace_to_method_in_allowed_methods(self):
        for method in UninitializedTensorMixin._allowed_methods:
            if method.__name__ == "to":
                self.assertFalse(hasattr(method, "__self__"))   # 替换后torch.Tensor.to变成普通函数，而不是原来的绑定方法
                break

    @patch('torch.distributed.Backend')
    def test_cuda_entry_exists(self, mock_backend):
        # 模拟 default_device_backend_map 的存在及其内容
        mock_backend.default_device_backend_map = {'cpu': 'gloo', 'cuda': 'nccl', 'npu': 'hccl'}

        _del_nccl_device_backend_map()

        # 验证 'cuda' 是否被删除
        self.assertNotIn('cuda', mock_backend.default_device_backend_map)

    @patch('torch.distributed.Backend')
    def test_cuda_entry_does_not_exist(self, mock_backend):
        # 模拟 default_device_backend_map 的存在但没有 'cuda'
        mock_backend.default_device_backend_map = {'cpu': 'gloo', 'npu': 'hccl'}

        _del_nccl_device_backend_map()

        # 验证 'cuda' 仍然不存在
        self.assertNotIn('cuda', mock_backend.default_device_backend_map)

    @patch('torch.distributed.Backend')
    def test_default_device_backend_map_not_exist(self, mock_backend):
        # 模拟 default_device_backend_map 不存在
        del mock_backend.default_device_backend_map

        _del_nccl_device_backend_map()

        # 没有抛出异常，测试通过


if __name__ == "__main__":
    run_tests()
