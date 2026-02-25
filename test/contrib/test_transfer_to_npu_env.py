import os

os.environ["TORCH_TRANSFER_TO_NPU"] = "1"

import torch
from torch.nn.parameter import UninitializedTensorMixin
from torch.utils.data import TensorDataset
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu


class TestTransferToNpuEnv(TestCase):
    """Test transfer_to_npu auto-import via environment variable"""

    def test_cuda_replaced_with_env(self):
        tensor = torch.randn(2, 3, device="cuda")
        self.assertEqual(tensor.device.type, 'npu')

        current_device = torch.cuda.current_device()
        self.assertIsInstance(current_device, int)

        stream = torch.cuda.current_stream()
        self.assertTrue(hasattr(stream, 'device'))

        stream = torch.cuda.Stream()
        self.assertTrue(hasattr(stream, 'device'))


class TestTransferToNpu(TestCase):

    def test_generator(self):
        g0 = torch.Generator()
        self.assertTrue(isinstance(g0, torch.Generator))
        self.assertEqual(g0.device.type, 'cpu')

        g1 = torch.Generator('cuda')
        self.assertTrue(isinstance(g1, torch.Generator))
        self.assertEqual(g1.device.type, 'npu')

        g2 = torch.Generator(torch.device('cuda'))
        self.assertTrue(isinstance(g2, torch.Generator))
        self.assertEqual(g2.device.type, 'npu')

        g3 = torch.Generator(device='cuda')
        self.assertTrue(isinstance(g3, torch.Generator))
        self.assertEqual(g3.device.type, 'npu')

        g4 = torch.Generator(device=torch.device('cuda'))
        self.assertTrue(isinstance(g4, torch.Generator))
        self.assertEqual(g4.device.type, 'npu')

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

        with torch.amp.autocast('cuda'):
            c = a @ b
        c_after_amp_autocast_args = c.dtype

        with torch.amp.autocast(device_type='cuda'):
            c = a @ b
        c_after_amp_autocast_kwargs = c.dtype

        self.assertNotEqual(c_before_autocast, c_after_amp_autocast_args)
        self.assertEqual(c_after_amp_autocast_args, c_after_amp_autocast_kwargs)

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
                self.assertFalse(hasattr(method, "__self__"))  # 替换后torch.Tensor.to变成普通函数，而不是原来的绑定方法
                break

    def test_torch_Event(self):
        event = torch.Event(device='cuda:0', enable_timing=True)
        self.assertEqual(str(event.device), 'npu')
        self.assertEqual(isinstance(event, torch.Event), True)

        event = torch.Event('cuda:0')
        self.assertEqual(str(event.device), 'npu')
        self.assertEqual(isinstance(event, torch.Event), True)

    def test_torch_get_device_module(self):
        device_module1 = torch.get_device_module(device='cuda:0')
        device_module2 = torch.get_device_module(device=torch.device('cuda:0'))
        npu_device_module = torch.get_device_module(device='npu:0')
        self.assertEqual(device_module1, npu_device_module)
        self.assertEqual(device_module2, npu_device_module)

    def test_torch_cuda_current_stream(self):
        cur_stream = torch.cuda.current_stream(0)
        cur_npu_stream = torch.npu.current_stream(0)
        self.assertEqual(cur_stream, cur_npu_stream)
        cur_stream = torch.cuda.current_stream(torch.device('cuda:0'))
        cur_npu_stream = torch.npu.current_stream(torch.device('npu:0'))
        self.assertEqual(cur_stream, cur_npu_stream)

    def test_torch_cuda_utilization(self):
        # 获取device利用率
        use = torch.cuda.utilization(0)
        use = torch.cuda.utilization(torch.device('cuda:0'))

    def test_torch_cuda_set_per_process_memory_fraction(self):
        torch.cuda.set_per_process_memory_fraction(0.5, device=0)
        torch.cuda.set_per_process_memory_fraction(1.0, device=torch.device('cuda:0'))

    def test_torch_cuda_caching_allocator_alloc(self):
        size = 1024 * 1024
        ptr1 = torch.cuda.caching_allocator_alloc(size, 0)
        ptr2 = torch.cuda.caching_allocator_alloc(size, torch.device('cuda:0'))
        torch.cuda.caching_allocator_delete(ptr1)
        torch.cuda.caching_allocator_delete(ptr2)

    def test_torch_cuda_memory(self):
        device = torch.device('cuda:0')
        a = torch.randn((1000, 1000), device=device)
        torch.cuda.memory._record_memory_history(device=device)
        snapshot = torch.cuda.memory._snapshot(device=device)

    def test_torch_fft(self):
        freq_fft = torch.fft.fftfreq(5, device=torch.device('cuda:0'))
        self.assertEqual(str(freq_fft.device), 'npu:0')
        freq_rfft = torch.fft.rfftfreq(5, device=torch.device('cuda:0'))
        self.assertEqual(str(freq_rfft.device), 'npu:0')

    def test_torch_autograd_profiler_util_Kernel(self):
        kernel = torch.autograd.profiler_util.Kernel("model_inference", 'cuda', 11)
        self.assertEqual(kernel.device, 'npu')

    def test_torch_utils_cpp_extension_include_paths(self):
        torch.utils.cpp_extension.include_paths(device_type='cuda')

    def test_init_process_group(self):
        MASTER_ADDR = "127.0.0.1"
        MASTER_PORT = "29500"
        WORLD_SIZE = 1
        RANK = 0

        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = MASTER_PORT
        os.environ['RANK'] = str(RANK)
        os.environ['WORLD_SIZE'] = str(WORLD_SIZE)

        try:

            torch.distributed.init_process_group(
                backend='nccl',
                init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
                world_size=WORLD_SIZE,
                rank=RANK,
                device_id=torch.device(f"cuda:{RANK}")
            )
            self.assertEqual(torch.distributed.get_backend(), 'hccl')
            torch.distributed.barrier()

        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    run_tests()
