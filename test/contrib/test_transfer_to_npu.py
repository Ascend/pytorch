import json
from unittest.mock import patch, mock_open

import torch
from torch.nn.parameter import UninitializedTensorMixin
from torch.utils.data import TensorDataset
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.contrib import transfer_to_npu


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

    @patch('torch.distributed.Backend')
    def test_cuda_entry_exists(self, mock_backend):
        # 模拟 default_device_backend_map 的存在及其内容
        mock_backend.default_device_backend_map = {'cpu': 'gloo', 'cuda': 'nccl', 'npu': 'hccl'}

        transfer_to_npu._del_nccl_device_backend_map()

        # 验证 'cuda' 是否被删除
        self.assertNotIn('cuda', mock_backend.default_device_backend_map)

    @patch('torch.distributed.Backend')
    def test_cuda_entry_does_not_exist(self, mock_backend):
        # 模拟 default_device_backend_map 的存在但没有 'cuda'
        mock_backend.default_device_backend_map = {'cpu': 'gloo', 'npu': 'hccl'}

        transfer_to_npu._del_nccl_device_backend_map()

        # 验证 'cuda' 仍然不存在
        self.assertNotIn('cuda', mock_backend.default_device_backend_map)

    @patch('torch.distributed.Backend')
    def test_default_device_backend_map_not_exist(self, mock_backend):
        # 模拟 default_device_backend_map 不存在
        del mock_backend.default_device_backend_map

        transfer_to_npu._del_nccl_device_backend_map()

        # 没有抛出异常，测试通过

    def test_input_validation(self):
        # Test file is a link
        with patch('os.path.islink', return_value=True):
            self.assertFalse(transfer_to_npu._check_input_file_valid('dummy_path'))

        # Test file does not exist
        with patch('os.path.islink', return_value=False), \
                patch('os.path.realpath'), \
                patch('os.path.exists', return_value=False):
            self.assertFalse(transfer_to_npu._check_input_file_valid('dummy_path'))

        # Test file not readable
        with patch('os.path.islink', return_value=False), \
                patch('os.path.realpath'), \
                patch('os.path.exists', return_value=True), \
                patch('os.access', return_value=False):
            self.assertFalse(transfer_to_npu._check_input_file_valid('dummy_path'))

        # Test file name too long
        with patch('os.path.islink', return_value=False), \
                patch('os.path.realpath'), \
                patch('os.path.exists', return_value=True), \
                patch('os.access', return_value=True), \
                patch('os.path.basename', return_value='a' * 201):
            self.assertFalse(transfer_to_npu._check_input_file_valid('dummy_path'))

        # Test file too large
        with patch('os.path.islink', return_value=False), \
                patch('os.path.realpath'), \
                patch('os.path.exists', return_value=True), \
                patch('os.access', return_value=True), \
                patch('os.path.basename', return_value='valid_name'), \
                patch('os.path.getsize', return_value=11 * 1024 ** 2):
            self.assertFalse(transfer_to_npu._check_input_file_valid('dummy_path'))

        # Test valid file
        with patch('os.path.islink', return_value=False), \
                patch('os.path.realpath'), \
                patch('os.path.exists', return_value=True), \
                patch('os.access', return_value=True), \
                patch('os.path.basename', return_value='valid_name'), \
                patch('os.path.getsize', return_value=1024):
            self.assertTrue(transfer_to_npu._check_input_file_valid('dummy_path'))

    def test_load_json_file(self):
        # Test with invalid file
        with patch('torch_npu.contrib.transfer_to_npu._check_input_file_valid', return_value=False):
            self.assertEqual(transfer_to_npu._load_json_file('invalid_path'), {})

        # Test with JSON decode error
        with patch('torch_npu.contrib.transfer_to_npu._check_input_file_valid', return_value=True), \
                patch('builtins.open', mock_open(read_data='invalid json')), \
                patch('json.load', side_effect=json.JSONDecodeError('Expecting value', 'doc', 0)):
            self.assertEqual(transfer_to_npu._load_json_file('dummy_path'), {})

        # Test with file content not a dict
        with patch('torch_npu.contrib.transfer_to_npu._check_input_file_valid', return_value=True), \
                patch('builtins.open', mock_open(read_data='["not", "a", "dict"]')), \
                patch('json.load', return_value=["not", "a", "dict"]):
            self.assertEqual(transfer_to_npu._load_json_file('dummy_path'), {})

        # Test with valid JSON dict
        valid_json_data = '{"key": "value"}'
        with patch('torch_npu.contrib.transfer_to_npu._check_input_file_valid', return_value=True), \
                patch('builtins.open', mock_open(read_data=valid_json_data)), \
                patch('json.load', return_value={"key": "value"}):
            self.assertEqual(transfer_to_npu._load_json_file('valid_path'), {"key": "value"})

    def test_wrapper_function(self):
        @transfer_to_npu._wrapper_libraries_func
        def test_function():
            return torch.cuda.is_available()

        self.assertFalse(test_function())

    def test_replace_cuda_to_npu_in_dict(self):
        input_dict = {
            "device": "cuda:0",
            "cuda_version": "10.2",
            "non_cuda_key": "no replacement needed",
            "123": "cuda_core"
        }
        expected_dict = {
            "device": "npu:0",
            "npu_version": "10.2",
            "non_npu_key": "no replacement needed",
            "123": "npu_core"
        }

        result_dict = transfer_to_npu._replace_cuda_to_npu_in_dict(input_dict)
        self.assertEqual(result_dict, expected_dict)

    def test_wrapper_hccl_args_and_kwargs(self):
        @transfer_to_npu._wrapper_hccl
        def mock_function(*args, **kwargs):
            return args, kwargs

        args_input = ('nccl', 'cpu')
        kwargs_input = {'backend': 'nccl', 'device': 'gpu'}
        expected_args_output = ('hccl', 'cpu')
        expected_kwargs_output = {'backend': 'hccl', 'device': 'gpu'}

        args_output, kwargs_output = mock_function(*args_input, **kwargs_input)

        self.assertEqual(args_output, expected_args_output)
        self.assertEqual(kwargs_output, expected_kwargs_output)

    def test_wrapper_profiler_experimental_config(self):
        @transfer_to_npu._wrapper_profiler
        def mock_function(*args, **kwargs):
            return kwargs

        wrong_config = 'not_a_valid_config'
        correct_config = torch_npu.profiler._ExperimentalConfig(1, 1)

        with patch('logging.warning') as mock_logger:
            result = mock_function(experimental_config=wrong_config)
            mock_logger.assert_called_once()
            self.assertNotIn('experimental_config', result)

        result = mock_function(experimental_config=correct_config)
        self.assertIn('experimental_config', result)
        self.assertIs(result['experimental_config'], correct_config)

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

    def test_torch_sparse_compressed_tensor(self):
        # 定义压缩的索引
        compressed_indices = torch.tensor([0, 3, 5], dtype=torch.int64)  # 表示行索引的压缩
        plain_indices = torch.tensor([[0, 0], [1, 2], [2, 3]], dtype=torch.int64)  # 非压缩索引
        values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)  # 非零值

        # 创建稀疏压缩矩阵
        sparse_tensor = torch.sparse_compressed_tensor(
            compressed_indices,
            plain_indices,
            values,
            size=(3, 4),
            dtype=torch.float32,
            layout=torch.sparse_csr,
            device=torch.device('cuda:0'),
            requires_grad=True
        )
        self.assertEqual(str(sparse_tensor.device), 'npu:0')

    def test_torch_utils_cpp_extension_include_paths(self):
        torch.utils.cpp_extension.include_paths(device_type='cuda')


if __name__ == "__main__":
    run_tests()
