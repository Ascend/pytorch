import os
import gc
import shutil
import threading
import subprocess

import torch
import torch.utils.cpp_extension
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase, run_tests, instantiate_parametrized_tests, parametrize
import torch_npu

PYTORCH_INSTALL_PATH = os.path.dirname(os.path.realpath(torch.__file__))
PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.realpath(torch_npu.__file__))
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"


def create_build_path(build_directory):
    if os.path.exists(build_directory):
        shutil.rmtree(build_directory, ignore_errors=True)
    os.makedirs(build_directory, exist_ok=True)


def build_stub(base_dir):
    build_stub_cmd = ["sh", os.path.join(base_dir, 'third_party/acl/libs/build_stub.sh')]
    if subprocess.call(build_stub_cmd) != 0:
        raise RuntimeError('Failed to build stub: {}'.format(build_stub_cmd))


class TestPluggableAllocator(TestCase):
    torch.npu.memory._set_allocator_settings("expandable_segments:True")
    module = None
    new_alloc = None
    build_directory = "allocator/build"
    conv = nn.Conv1d(1024, 256, 4, stride=4).to("npu")
    deconv = nn.ConvTranspose1d(256, 1024, 4, stride=4).to("npu")

    @classmethod
    def setUpClass(cls):
        os_path = os.path.join(cls.build_directory, 'pluggable_allocator_extensions.so')
        if os.path.exists(os_path):
            cls.new_alloc = torch_npu.npu.memory.NPUPluggableAllocator(os_path, 'my_malloc', 'my_free')
            return

        # Build Extension
        BASE_DIR = os.path.abspath("./../")
        build_stub(BASE_DIR)
        create_build_path(cls.build_directory)
        CANN_LIB_PATH = os.path.join(BASE_DIR, 'third_party/acl/libs')
        extra_ldflags = []
        extra_ldflags.append("-lascendcl")
        extra_ldflags.append(f"-L{CANN_LIB_PATH}")
        extra_ldflags.append("-lc10")
        extra_ldflags.append(f"-L{PYTORCH_INSTALL_PATH}")
        extra_include_paths = ["cpp_extensions"]
        extra_include_paths.append(os.path.join(PYTORCH_NPU_INSTALL_PATH, 'include'))

        cls.module = torch.utils.cpp_extension.load(
            name="pluggable_allocator_extensions",
            sources=[
                "cpp_extensions/pluggable_allocator_extensions.cpp"
            ],
            extra_include_paths=extra_include_paths,
            extra_cflags=["-g"],
            extra_ldflags=extra_ldflags,
            build_directory=cls.build_directory,
            verbose=True,
        )
        # Load the allocator
        cls.new_alloc = torch_npu.npu.memory.NPUPluggableAllocator(os_path, 'my_malloc', 'my_free')
    
    def test_pluggable_allocator(self):
        torch.npu.memory._set_allocator_settings("expandable_segments:False")
        with torch.npu.use_mem_pool(torch.npu.MemPool(TestPluggableAllocator.new_alloc._allocator)):
            x = torch.empty((7500, 1024, 1024), device="npu")
            del x
        torch.npu.memory._set_allocator_settings("expandable_segments:True")
    
    @staticmethod
    def conv_operation(x):
        return TestPluggableAllocator.deconv(TestPluggableAllocator.conv(x) + 0.005)

    @staticmethod
    def conv_with_allocator(x):
        torch.npu.memory._set_allocator_settings("expandable_segments:False")
        with torch.npu.use_mem_pool(torch.npu.MemPool(TestPluggableAllocator.new_alloc._allocator)):
            x = TestPluggableAllocator.conv_operation(x)
        torch.npu.memory._set_allocator_settings("expandable_segments:True")
        return x

    @parametrize("task_queue_enable", [0, 1, 2])
    def test_task_queue(self, task_queue_enable):
        os.environ["TASK_QUEUE_ENABLE"] = str(task_queue_enable)
        input_data = torch.randn(1, 1024, 96, dtype=torch.float32, device="npu")
        x1 = input_data
        for _ in range(5):
            x1 = self.conv_operation(x1)
            x1 = self.conv_with_allocator(x1)
        x2 = input_data
        for _ in range(10):
            x2 = self.conv_operation(x2)
        self.assertEqual(x1, x2)
        os.environ["TASK_QUEUE_ENABLE"] = "1"

    def test_thread_share(self):
        lock = threading.Lock()

        def worker(name, shared_tensor):
            torch.npu.synchronize()
            with lock:
                shared_tensor.sub_(1)
            torch.npu.synchronize()
        torch.npu.memory._set_allocator_settings("expandable_segments:False")
        with torch.npu.use_mem_pool(torch.npu.MemPool(TestPluggableAllocator.new_alloc._allocator)):
            input_data = torch.zeros((4, 4), dtype=torch.float32, device="npu")
            with lock:
                input_data.add_(1)
            th = threading.Thread(target=worker, args=("thread1", input_data))
            th.start()
            th.join()
        self.assertEqual(input_data, torch.zeros((4, 4), dtype=torch.float32, device="npu"))

    def test_mul_stream(self):
        input_data = torch.randn(1, 1024, 96, dtype=torch.float32, device="npu")
        x1 = input_data
        x2 = input_data
        stream1, stream2 = torch.npu.Stream(), torch.npu.Stream()
        events = [torch.npu.Event(False, False) for _ in range(3)]

        with torch.npu.stream(stream1):
            x1 = self.conv_with_allocator(x1)
            events[0].record()
        
        with torch.npu.stream(stream2):
            events[0].wait(stream2)
            x2 = self.conv_operation(x2)
            events[1].record()
        
        with torch.npu.stream(stream1):
            events[1].wait(stream1)
            x1 = self.conv_with_allocator(x1)
            events[2].record()
        
        with torch.npu.stream(stream2):
            events[2].wait(stream2)
            x2 = self.conv_operation(x2)

        torch.npu.synchronize()
        self.assertEqual(x1, x2)
    
    def test_mul_stream_with_threads(self):
        input_data = torch.randn(1, 1024, 96, dtype=torch.float32, device="npu")
        events = [torch.npu.Event(False, False) for _ in range(3)]
        
        def stream_worker(data, stream, event_sequence):
            """Generic stream worker function"""
            with torch.npu.stream(stream):
                for event, operation in event_sequence:
                    event.wait(stream)
                    data = operation(data)
                    events[event_sequence.index((event, operation)) + 1].record()
            return data
        
        # Define operation sequences for two streams
        stream1_ops = [
            (events[0], self.conv_with_allocator),
            (events[1], self.conv_operation)
        ]
        stream2_ops = [
            (events[0], self.conv_operation),
            (events[2], self.conv_with_allocator)
        ]

        result_container = {}
        stream2 = torch.npu.Stream()
        
        def thread_func():
            result_container["x2"] = stream_worker(input_data, stream2, stream2_ops)

        thread = threading.Thread(target=thread_func)
        thread.start()

        stream1 = torch.npu.Stream()
        x1 = stream_worker(input_data, stream1, stream1_ops)

        thread.join()
        torch.npu.synchronize()

        self.assertEqual(x1, result_container["x2"])

    def test_dict_data_loader(self):
        class DictDataset(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                torch.npu.memory._set_allocator_settings("expandable_segments:False")
                with torch.npu.use_mem_pool(torch.npu.MemPool(TestPluggableAllocator.new_alloc._allocator)):
                    ret_dict = {
                        "a_tensor": torch.randn(4, 2, dtype=torch.float32, device="npu"),
                        "another_dict": {"a_number": idx}
                    }
                torch.npu.memory._set_allocator_settings("expandable_segments:True")
                return ret_dict

        class TestDictDataLoader():
            def __init__(self):
                self.dataset = DictDataset()
            
            def test_memory(self):
                loader = DataLoader(self.dataset, batch_size=2)
                for sample in loader:
                    print(f'sample: {sample}')

        loader = TestDictDataLoader()
        loader.test_memory()

instantiate_parametrized_tests(TestPluggableAllocator)

if __name__ == '__main__':
    run_tests()
