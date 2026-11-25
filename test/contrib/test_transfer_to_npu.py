import os
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
        try:
            torch.set_default_device("cuda")
            a = torch.tensor(1)
            self.assertEqual(a.device.type, 'npu')
        finally:
            torch.set_default_device(None)

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

    def test_stream_record_event(self):
        # torch.Stream.record_event rejects subclass instances via C++ exact
        # type matching; _EventProxy must return a base torch.Event to pass it.
        event = torch.Event(device='cuda:0')
        stream = torch.Stream(torch.device('npu'))
        recorded = stream.record_event(event)
        self.assertEqual(isinstance(recorded, torch.Event), True)
        self.assertTrue('npu' in str(recorded.device))

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
        positional_cases = (
            ("cuda_kernel", "cuda:0", 11.5, "npu:0"),
            ("", "", 0, ""),
            ("cuda_kernel", torch.device("cuda:2"), 12, torch.device("npu:2")),
            ("cuda_kernel", 0, 0, 0),
            ("cuda_kernel", None, 1, None),
            ("cuda_kernel", ["cuda:0", 0], 1, ["cuda:0", 0]),
            ("cuda_kernel", ("cuda:0", 0), 1, ("cuda:0", 0)),
            ("cuda_kernel", {}, 0, {}),
            ("cuda_kernel", {"cuda:0": "cuda:1"}, 13, {"npu:0": "npu:1"}),
        )
        for name, device, duration, expected_device in positional_cases:
            with self.subTest(name=name, device=device, duration=duration):
                kernel = torch.autograd.profiler_util.Kernel(name, device, duration)
                self.assertEqual(kernel.name, name)
                self.assertEqual(kernel.device, expected_device)
                self.assertEqual(kernel.duration, duration)

        kernel = torch.autograd.profiler_util.Kernel(
            name="cuda_kernel", device="cuda:1", duration=14
        )
        self.assertEqual(kernel.name, "cuda_kernel")
        self.assertEqual(kernel.device, "npu:1")
        self.assertEqual(kernel.duration, 14)

        with self.assertRaises(TypeError):
            torch.autograd.profiler_util.Kernel("cuda_kernel", "cuda:0")
        with self.assertRaises(TypeError):
            torch.autograd.profiler_util.Kernel("cuda_kernel", "cuda:0", 1, 2)
        with self.assertRaises(TypeError):
            torch.autograd.profiler_util.Kernel(
                name="cuda_kernel", device="cuda:0", duration=1, unknown=True
            )

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
                backend='cuda:nccl',
                init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
                world_size=WORLD_SIZE,
                rank=RANK,
                device_id=torch.device(f"cuda:{RANK}")
            )
            self.assertEqual(torch.distributed.get_backend(), 'npu:hccl')
            torch.distributed.barrier()

        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

    def test_process_group_nccl(self):
        from torch.distributed import ProcessGroupNCCL
        pg_options = ProcessGroupNCCL.Options()
        self.assertIsNotNone(pg_options)

        from torch.distributed.distributed_c10d import ProcessGroupNCCL
        pg_options = ProcessGroupNCCL.Options()
        self.assertIsNotNone(pg_options)

    def test_host_empty_cache_is_patched(self):
        self.assertEqual(torch._C._host_emptyCache, torch_npu._C._npu_hostEmptyCache)

    def test_update_cuda_default_generators(self):
        torch.randn(1).npu()
        self.assertEqual(torch.cuda.default_generators, torch_npu.npu.default_generators)
        self.assertNotEqual(torch.cuda.default_generators, ())

    def test_wrapper_cuda_device_ids_tuple(self):
        @transfer_to_npu._wrapper_cuda
        def mock_function(*args, **kwargs):
            return kwargs

        kwargs_input = {'device_ids': ('cuda:0', 'cuda:1')}
        expected_kwargs_output = {'device_ids': ('npu:0', 'npu:1')}
        kwargs_output = mock_function(**kwargs_input)
        self.assertEqual(kwargs_output, expected_kwargs_output)

        kwargs_input = {'device_ids': ['cuda:0', 'cuda:1']}
        expected_kwargs_output = {'device_ids': ['npu:0', 'npu:1']}
        kwargs_output = mock_function(**kwargs_input)
        self.assertEqual(kwargs_output, expected_kwargs_output)

    def test_is_autocast_enabled(self):
        with torch.autocast('cuda'):
            self.assertTrue(torch.is_autocast_enabled())
            self.assertTrue(torch.is_autocast_enabled('cuda'))
            self.assertTrue(torch.is_autocast_enabled('cuda:0'))
            self.assertTrue(torch.is_autocast_enabled('npu'))
            self.assertFalse(torch.is_autocast_enabled('cpu'))

        self.assertFalse(torch.is_autocast_enabled())
        self.assertFalse(torch.is_autocast_enabled('cuda'))
        self.assertFalse(torch.is_autocast_enabled('npu'))

    def test_amp_autocast_cuda_redirected(self):
        with torch.amp.autocast('cuda'):
            self.assertTrue(torch.is_autocast_enabled('npu'))

    def test_custom_fwd_bwd_cuda_device_type(self):
        captured = {}

        class Fn(torch.autograd.Function):
            @staticmethod
            @torch.amp.custom_fwd(device_type='cuda')
            def forward(ctx, x):
                captured['fwd_used_autocast'] = ctx._fwd_used_autocast
                return x * 2

            @staticmethod
            @torch.amp.custom_bwd(device_type='cuda')
            def backward(ctx, grad_output):
                captured['bwd_autocast'] = torch.is_autocast_enabled('npu')
                return grad_output * 2

        with torch.autocast('npu'):
            x = torch.ones(1, device='npu', requires_grad=True)
            y = Fn.apply(x)
            y.backward()

        self.assertTrue(captured['fwd_used_autocast'])
        self.assertTrue(captured['bwd_autocast'])

    def test_custom_fwd_bwd_cuda_with_index(self):
        captured = {}

        class Fn(torch.autograd.Function):
            @staticmethod
            @torch.amp.custom_fwd(device_type='cuda:0')
            def forward(ctx, x):
                captured['fwd_used_autocast'] = ctx._fwd_used_autocast
                return x * 2

            @staticmethod
            @torch.amp.custom_bwd(device_type='cuda:0')
            def backward(ctx, grad_output):
                captured['bwd_autocast'] = torch.is_autocast_enabled('npu')
                return grad_output * 2

        with torch.autocast('npu'):
            x = torch.ones(1, device='npu', requires_grad=True)
            Fn.apply(x).backward()

        self.assertTrue(captured['fwd_used_autocast'])
        self.assertTrue(captured['bwd_autocast'])

    def test_get_autocast_dtype(self):
        with torch.autocast('npu', torch.bfloat16):
            self.assertEqual(torch.get_autocast_gpu_dtype(), torch.bfloat16)
            self.assertEqual(torch.get_autocast_dtype('cuda'), torch.bfloat16)
            self.assertEqual(torch.get_autocast_dtype('npu'), torch.bfloat16)
            self.assertEqual(torch_npu.npu.get_autocast_dtype(), torch.bfloat16)

        self.assertEqual(torch.get_autocast_gpu_dtype(), torch.float16)
        self.assertEqual(torch.get_autocast_dtype('cuda'), torch.float16)
        self.assertEqual(torch.get_autocast_dtype('npu'), torch.float16)
        self.assertEqual(torch_npu.npu.get_autocast_dtype(), torch.float16)

    def test_set_autocast_enabled_no_device_arg(self):
        # set_autocast_enabled without device_type should redirect to NPU
        torch.set_autocast_enabled(True)
        self.assertTrue(torch.is_autocast_enabled())
        self.assertTrue(torch.is_autocast_enabled('npu'))
        self.assertTrue(torch.is_autocast_enabled('cuda'))

        torch.set_autocast_enabled(False)
        self.assertFalse(torch.is_autocast_enabled())
        self.assertFalse(torch.is_autocast_enabled('npu'))

        # keyword arg
        torch.set_autocast_enabled(enabled=True)
        self.assertTrue(torch.is_autocast_enabled('npu'))
        torch.set_autocast_enabled(enabled=False)
        self.assertFalse(torch.is_autocast_enabled('npu'))

    def test_set_autocast_enabled_dtype(self):
        # Inside autocast: NPU enabled, dtype bfloat16
        with torch.autocast('npu', torch.bfloat16):
            self.assertTrue(torch.is_autocast_enabled('npu'))
            self.assertEqual(torch.get_autocast_dtype('npu'), torch.bfloat16)

            # set_autocast_enabled('cuda', False) should redirect to NPU
            torch.set_autocast_enabled('cuda', False)
            self.assertFalse(torch.is_autocast_enabled('npu'))
            self.assertFalse(torch.is_autocast_enabled('cuda'))

            # set_autocast_enabled('cuda', True) should restore NPU
            torch.set_autocast_enabled('cuda', True)
            self.assertTrue(torch.is_autocast_enabled('npu'))

            # set_autocast_dtype('cuda', float16) should redirect to NPU
            torch.set_autocast_dtype('cuda', torch.float16)
            self.assertEqual(torch.get_autocast_dtype('npu'), torch.float16)

            # set_autocast_gpu_dtype should redirect to NPU
            torch.set_autocast_gpu_dtype(torch.bfloat16)
            self.assertEqual(torch.get_autocast_gpu_dtype(), torch.bfloat16)
            self.assertEqual(torch.get_autocast_dtype('npu'), torch.bfloat16)

        # Outside autocast: verify setter/getter consistency
        torch.set_autocast_enabled('cuda', True)
        self.assertTrue(torch.is_autocast_enabled('npu'))
        self.assertTrue(torch.is_autocast_enabled('cuda'))
        torch.set_autocast_enabled('cuda', False)

        torch.set_autocast_dtype('cuda', torch.bfloat16)
        self.assertEqual(torch.get_autocast_dtype('npu'), torch.bfloat16)
        torch.set_autocast_dtype('cuda', torch.float16)

        torch.set_autocast_gpu_dtype(torch.bfloat16)
        self.assertEqual(torch.get_autocast_gpu_dtype(), torch.bfloat16)
        torch.set_autocast_gpu_dtype(torch.float16)

    def test_cuda_memory_submodule_redirect(self):
        # _patch_cuda 新增 cuda.memory 路径，同名接口应指向 torch_npu.npu.memory.*
        self.assertIs(torch.cuda.memory.memory_allocated, torch_npu.npu.memory.memory_allocated)
        self.assertIs(torch.cuda.memory.memory_cached, torch_npu.npu.memory.memory_cached)
        self.assertIs(torch.cuda.memory.memory_reserved, torch_npu.npu.memory.memory_reserved)
        self.assertIs(torch.cuda.memory.max_memory_allocated, torch_npu.npu.memory.max_memory_allocated)
        self.assertIs(torch.cuda.memory.max_memory_cached, torch_npu.npu.memory.max_memory_cached)
        self.assertIs(torch.cuda.memory.max_memory_reserved, torch_npu.npu.memory.max_memory_reserved)
        self.assertIs(torch.cuda.memory.empty_cache, torch_npu.npu.memory.empty_cache)
        self.assertIs(torch.cuda.memory.memory_stats, torch_npu.npu.memory.memory_stats)
        self.assertIs(torch.cuda.memory.memory_stats_as_nested_dict, torch_npu.npu.memory.memory_stats_as_nested_dict)
        self.assertIs(torch.cuda.memory.memory_snapshot, torch_npu.npu.memory.memory_snapshot)
        self.assertIs(torch.cuda.memory.memory_summary, torch_npu.npu.memory.memory_summary)
        self.assertIs(torch.cuda.memory.get_allocator_backend, torch_npu.npu.memory.get_allocator_backend)
        self.assertIs(torch.cuda.memory.change_current_allocator, torch_npu.npu.memory.change_current_allocator)
        self.assertIs(torch.cuda.memory.MemPool, torch_npu.npu.memory.MemPool)
        self.assertIs(torch.cuda.memory.MemPoolContext, torch_npu.npu.memory.MemPoolContext)
        self.assertIs(torch.cuda.memory.use_mem_pool, torch_npu.npu.memory.use_mem_pool)

    def test_cuda_memory_explicit_redirect(self):
        # 命名不同或不在 __all__ 的接口，须显式赋值
        self.assertIs(torch.cuda.memory.CUDAPluggableAllocator, torch_npu.npu.memory.NPUPluggableAllocator)
        self.assertIs(torch.cuda.memory.is_initialized, torch_npu.npu.memory.is_initialized)

    def test_cuda_memory_runtime(self):
        device = torch.device('cuda:0')
        a = torch.randn((100, 100), device=device)
        self.assertGreater(torch.cuda.memory.memory_allocated(), 0)
        self.assertGreaterEqual(torch.cuda.memory.memory_cached(), 0)
        self.assertGreaterEqual(torch.cuda.memory.memory_reserved(), 0)
        self.assertTrue(torch.cuda.memory.is_initialized())
        self.assertEqual(torch.cuda.memory.get_allocator_backend(), 'native')
        self.assertIsInstance(torch.cuda.memory.memory_stats(), dict)
        self.assertIsInstance(torch.cuda.memory.memory_stats_as_nested_dict(), dict)
        self.assertIsInstance(torch.cuda.memory.host_memory_stats(), dict)
        self.assertIsInstance(torch.cuda.memory.host_memory_stats_as_nested_dict(), dict)
        torch.cuda.memory.empty_cache()
        torch.cuda.memory.reset_accumulated_memory_stats()
        torch.cuda.memory.reset_accumulated_host_memory_stats()
        torch.cuda.memory.reset_peak_memory_stats()
        torch.cuda.memory.reset_peak_host_memory_stats()
        torch.cuda.memory.reset_max_memory_allocated()
        torch.cuda.memory.reset_max_memory_cached()
        torch.cuda.memory.set_per_process_memory_fraction(0.5)
        self.assertEqual(torch.cuda.memory.get_per_process_memory_fraction(), 0.5)

    def test_cuda_memory_caching_allocator(self):
        size = 1024 * 1024
        ptr = torch.cuda.memory.caching_allocator_alloc(size, 0)
        torch.cuda.memory.caching_allocator_delete(ptr)

    def test_cuda_memory_caching_allocator_enable(self):
        # 回归：is_initialized 被重定向为 NPU 实现后，torch 的 CUDA-only
        # caching_allocator_enable 会调用 torch._C 中不存在的
        # _cuda_cudaCachingAllocator_enable，须提供 NPU 侧兼容实现。
        torch.randn(4, device='npu')
        self.assertTrue(torch.cuda.memory.is_initialized())
        # torch.cuda.caching_allocator_enable 是 torch.cuda.memory 中函数的独立
        # 别名，两处绑定都须替换。
        self.assertIs(torch.cuda.caching_allocator_enable,
                      torch.cuda.memory.caching_allocator_enable)
        torch.cuda.caching_allocator_enable(False)
        torch.cuda.caching_allocator_enable(True)

    def test_cuda_streams_submodule_redirect(self):
        self.assertIs(torch.cuda.streams.Stream, torch_npu.npu.streams.Stream)
        self.assertIs(torch.cuda.streams.Event, torch_npu.npu.streams.Event)
        self.assertIs(torch.cuda.streams.ExternalStream, torch_npu.npu.streams.ExternalStream)
        self.assertIs(torch.cuda.streams.ExternalEvent, torch_npu.npu.streams.ExternalEvent)
        stream = torch.cuda.streams.Stream()
        event = torch.cuda.streams.Event()
        self.assertIsInstance(stream, torch_npu.npu.streams.Stream)
        self.assertIsInstance(event, torch_npu.npu.streams.Event)

    def test_cuda_graphs_submodule_redirect(self):
        self.assertIs(torch.cuda.graphs.CUDAGraph, torch_npu.npu.graphs.NPUGraph)
        self.assertIs(torch.cuda.graphs.graph, torch_npu.npu.graphs.graph)
        self.assertIs(torch.cuda.graphs.graph_pool_handle, torch_npu.npu.graphs.graph_pool_handle)
        self.assertIs(torch.cuda.graphs.make_graphed_callables, torch_npu.npu.graphs.make_graphed_callables)
        self.assertFalse(torch.cuda.graphs.is_current_stream_capturing())

    def test_cuda_random_submodule_redirect(self):
        self.assertIs(torch.cuda.random.current_device, torch_npu.npu.current_device)
        self.assertIs(torch.cuda.random.device_count, torch_npu.npu.device_count)
        self.assertEqual(torch.cuda.random.current_device(), 0)
        self.assertEqual(torch.cuda.random.device_count(), torch_npu.npu.device_count())

    def test_grad_scaler_proxy(self):
        # torch.amp.GradScaler / torch.GradScaler 默认 device='cuda'，须透传为 npu GradScaler
        self.assertIs(torch.GradScaler, torch.amp.GradScaler)
        scaler = torch.amp.GradScaler(device='cuda')
        self.assertTrue(scaler.is_enabled())
        scaler = torch.GradScaler(device='cuda', init_scale=2.0)
        self.assertEqual(scaler.get_scale(), 2.0)

    def test_cuda_random_rng_state(self):
        # torch.cuda.random.get/set_rng_state 走 npu rng
        state = torch.cuda.random.get_rng_state()
        self.assertIsInstance(state, torch.Tensor)
        torch.cuda.random.set_rng_state(state)

    def test_fork_rng(self):
        # torch.random.fork_rng(device_type='cuda') 走 npu rng
        with torch.random.fork_rng(device_type='cuda'):
            torch.randn(1, device='npu')

    def test_save_on_cpu(self):
        # torch.autograd.graph.save_on_cpu(device_type='cuda') 走 npu
        torch.autograd.graph.save_on_cpu(pin_memory=True, device_type='cuda')

    def test_checkpoint_set_device_states(self):
        # torch.utils.checkpoint.set_device_states(device_type='cuda') 走 npu
        torch.utils.checkpoint.set_device_states([], [], device_type='cuda')

    def test_typed_storage(self):
        # Storage 构造器 TypedStorage.__new__ 的 device='cuda' 须重写为 npu
        s = torch.storage.TypedStorage(1, device='cuda')
        self.assertEqual(s.device.type, 'npu')
        self.assertTrue(s.is_cuda)

    def test_torch_stream(self):
        # torch.Stream 代理：device='cuda' 须重写为 npu
        s = torch.Stream(device='cuda')
        self.assertEqual(s.device.type, 'npu')
        self.assertIsInstance(s, torch.Stream)
        # 代理类须保持与原生类的子类关系：Dynamo 通过 issubclass(value, torch.Stream)
        # 判定是否按 StreamVariable 构造，缺失会导致 graph break 及后续编译错误。
        self.assertTrue(issubclass(torch_npu.npu.streams.Stream, torch.Stream))
        self.assertTrue(issubclass(torch_npu.npu.streams.Event, torch.Event))

    def test_torch_compile_cuda_stream_context(self):
        # 回归：torch.Stream 被重绑为 _StreamProxy 后，torch.cuda.Stream 生成的
        # npu Stream 仍须命中 issubclass 分支，torch.cuda.stream(...) 建图时不得
        # graph break，也不能把半初始化的 StreamContext 交给 Dynamo 编译。
        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()

            with torch.cuda.stream(s1):
                a = x + 1
            e = s1.record_event()
            s2.wait_event(e)
            with torch.cuda.stream(s2):
                b = a * 2

            s1.synchronize()
            s2.synchronize()
            return b

        x = torch.randn(1024, device='cuda')
        expected = fn(x)
        # Dynamo 的 VariableBuilder 会调用 torch.cuda.get_device_capability() 判断
        # Triton TMA 能力；torch_npu 在未配置该环境变量时返回 None，会在本用例要
        # 验证的逻辑之前抛出 TypeError。这里按 NPU 兼容约定临时配置并在结束后还原。
        capability_env = 'TORCH_NPU_DEVICE_CAPABILITY'
        saved_capability = os.environ.get(capability_env)
        if saved_capability is None:
            os.environ[capability_env] = '8.0'
        try:
            result = torch.compile(fn, backend='eager', fullgraph=True)(x)
        finally:
            if saved_capability is None:
                os.environ.pop(capability_env, None)
            else:
                os.environ[capability_env] = saved_capability
        self.assertEqual(result, expected)

if __name__ == "__main__":
    run_tests()
