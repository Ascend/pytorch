import os
import shutil
import threading
import subprocess
from types import SimpleNamespace

import torch
import torch.utils.cpp_extension
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase, run_tests, instantiate_parametrized_tests, parametrize, serialTest
import torch_npu

PYTORCH_INSTALL_PATH = os.path.dirname(os.path.realpath(torch.__file__))
PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.realpath(torch_npu.__file__))

def create_build_path(build_directory):
    if os.path.exists(build_directory):
        shutil.rmtree(build_directory, ignore_errors=True)
    os.makedirs(build_directory, exist_ok=True)


def build_stub(base_dir):
    build_stub_cmd = ["sh", os.path.join(base_dir, 'third_party/acl/libs/build_stub.sh')]
    if subprocess.call(build_stub_cmd) != 0:
        raise RuntimeError('Failed to build stub: {}'.format(build_stub_cmd))


def _load_module_from_so(os_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pluggable_allocator_extensions", os_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_pluggable_allocator_cache = None


def get_pluggable_allocator():
    """Build/load the pluggable allocator C++ extension.

    Returns a SimpleNamespace with:
    - module: the loaded Python module (or None if build failed)
    - build_directory: path to the build directory containing the .so
    """
    global _pluggable_allocator_cache
    if _pluggable_allocator_cache is not None:
        return _pluggable_allocator_cache

    build_directory = "allocator/build"
    os_path = os.path.join(build_directory, 'pluggable_allocator_extensions.so')

    module = None
    if os.path.exists(os_path):
        module = _load_module_from_so(os_path)
    else:
        TEST_DIR = os.path.dirname(os.path.abspath(__file__))
        BASE_DIR = os.path.dirname(TEST_DIR)
        build_stub(BASE_DIR)
        create_build_path(build_directory)
        CANN_LIB_PATH = os.path.join(BASE_DIR, 'third_party/acl/libs')
        extra_ldflags = []
        extra_ldflags.append("-lascendcl")
        extra_ldflags.append(f"-L{CANN_LIB_PATH}")
        extra_ldflags.append("-lc10")
        extra_ldflags.append(f"-L{PYTORCH_INSTALL_PATH}")
        extra_include_paths = [os.path.join(TEST_DIR, "cpp_extensions")]
        extra_include_paths.append(os.path.join(PYTORCH_NPU_INSTALL_PATH, 'include'))
        extra_include_paths.append(os.path.join(PYTORCH_NPU_INSTALL_PATH, 'include', 'third_party', 'acl', 'inc'))
        module = torch.utils.cpp_extension.load(
            name="pluggable_allocator_extensions",
            sources=[
                os.path.join(TEST_DIR, "cpp_extensions", "pluggable_allocator_extensions.cpp")
            ],
            extra_include_paths=extra_include_paths,
            extra_cflags=["-g"],
            extra_ldflags=extra_ldflags,
            build_directory=build_directory,
            verbose=True,
        )

    _pluggable_allocator_cache = SimpleNamespace(
        module=module,
        build_directory=build_directory,
    )
    return _pluggable_allocator_cache


class TestPluggableAllocator(TestCase):
    module = None
    new_alloc = None
    build_directory = "allocator/build"
    conv = nn.Conv1d(1024, 256, 4, stride=4).to("npu")
    deconv = nn.ConvTranspose1d(256, 1024, 4, stride=4).to("npu")
    _saved_allocator_settings = None

    @classmethod
    def setUpClass(cls):
        cls._saved_allocator_settings = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
        torch.npu.memory._set_allocator_settings("expandable_segments:False")
        pa = get_pluggable_allocator()
        cls.module = pa.module
        cls.build_directory = pa.build_directory
        if cls.module is not None:
            os_path = os.path.join(cls.build_directory, 'pluggable_allocator_extensions.so')
            cls.new_alloc = torch_npu.npu.memory.NPUPluggableAllocator(os_path, 'my_malloc', 'my_free')

    @classmethod
    def tearDownClass(cls):
        torch.npu.memory._set_allocator_settings(cls._saved_allocator_settings)

    def test_pluggable_allocator(self):
        with torch.npu.use_mem_pool(torch.npu.MemPool(TestPluggableAllocator.new_alloc._allocator)):
            x = torch.empty((7500, 1024, 1024), device="npu")
            del x

    @serialTest()
    def test_custom_allocator_alloc_free_paired(self):
        # Core regression for the mempool ownership rewrite: blocks allocated by a
        # custom allocator must be freed by the same allocator, even after the
        # use_mem_pool context exited and no pool is active anymore.
        if TestPluggableAllocator.module is None:
            self.skipTest("pluggable allocator module not available")
        TestPluggableAllocator.module.reset_alloc_free_count()
        try:
            pool = torch.npu.MemPool(TestPluggableAllocator.new_alloc._allocator)
            with torch.npu.use_mem_pool(pool):
                x = torch.empty((1024, 1024), device="npu")
            # Free the tensor outside the pool context, so there is no active pool.
            del x
            # Drop the pool, which releases the pool and empties its cache, forcing
            # the custom allocator's free path to run.
            del pool
            torch.npu.synchronize()
            torch.npu.empty_cache()

            alloc_count = TestPluggableAllocator.module.get_alloc_count()
            free_count = TestPluggableAllocator.module.get_free_count()
            self.assertGreater(alloc_count, 0)
            self.assertGreater(free_count, 0)
            self.assertEqual(alloc_count, free_count)
        finally:
            TestPluggableAllocator.module.reset_alloc_free_count()

    @staticmethod
    def conv_operation(x):
        return TestPluggableAllocator.deconv(TestPluggableAllocator.conv(x) + 0.005)

    @staticmethod
    def conv_with_allocator(x):
        with torch.npu.use_mem_pool(torch.npu.MemPool(TestPluggableAllocator.new_alloc._allocator)):
            x = TestPluggableAllocator.conv_operation(x)
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
                with torch.npu.use_mem_pool(torch.npu.MemPool(TestPluggableAllocator.new_alloc._allocator)):
                    ret_dict = {
                        "a_tensor": torch.randn(4, 2, dtype=torch.float32, device="npu"),
                        "another_dict": {"a_number": idx}
                    }
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


class TestMemPool(TestCase):
    _saved_allocator_settings = None

    @classmethod
    def setUpClass(cls):
        cls._saved_allocator_settings = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
        torch.npu.memory._set_allocator_settings("expandable_segments:False")

    @classmethod
    def tearDownClass(cls):
        torch.npu.memory._set_allocator_settings(cls._saved_allocator_settings)

    def test_mempool_id(self):
        pool1 = torch_npu.npu.graph_pool_handle()
        pool2 = torch_npu.npu.MemPool().id

        # first value of id in a user created pool is always zero
        self.assertEqual(pool1[0] == 0, pool2[0] == 0)

        # each call to torch_npu.npu.graph_pool_handle() or torch_npu.npu.MemPool()
        # increments the id
        self.assertTrue(abs(pool2[1] - pool1[1]) > 0)

    def test_mempool_public_api(self):
        # Aligned with PyTorch main: MemPool exposes id/use_count, and the old
        # MemPoolContext / MemPool.allocator surfaces are removed.
        pool = torch_npu.npu.MemPool()
        self.assertIsInstance(pool.id, tuple)
        self.assertEqual(len(pool.id), 2)

        # pool's use count should be 1 at this point as MemPool object
        # holds a reference
        self.assertEqual(pool.use_count(), 1)
        with torch_npu.npu.use_mem_pool(pool):
            # use_mem_pool takes a reference, bumping use_count to 2
            self.assertEqual(pool.use_count(), 2)
        # context exit drops the reference, use_count back to 1
        self.assertEqual(pool.use_count(), 1)
        del pool

    def test_mempool_multithread(self):
        pool_ids = []

        def create_mempool():
            pool_ids.append(torch_npu.npu.MemPool().id)

        num_threads = 4
        threads = [threading.Thread(target=create_mempool) for _ in range(num_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # each thread should create a unique mempool, since
        # mempool id creation is atomic
        self.assertEqual(len(set(pool_ids)), 4)

    @serialTest()
    def test_tensor_delete_after_allocator_delete(self):
        # Mirrors PyTorch's test_tensor_delete_after_allocator_delete: when the
        # allocator is dropped before the tensor, the PrivatePool instance must
        # keep the allocator live until all tensors die.
        pa = get_pluggable_allocator()
        if pa.module is None:
            self.skipTest("pluggable allocator module not available")
        so_path = os.path.join(
            pa.build_directory,
            'pluggable_allocator_extensions.so')
        allocator = torch_npu.npu.memory.NPUPluggableAllocator(so_path, 'my_malloc', 'my_free')
        pool = torch_npu.npu.MemPool(allocator._allocator)

        pa.module.reset_alloc_free_count()
        try:
            with torch_npu.npu.use_mem_pool(pool):
                data = torch.empty(4, device="npu")

            alloc_count = pa.module.get_alloc_count()
            free_count = pa.module.get_free_count()
            self.assertGreater(alloc_count, 0)
            self.assertEqual(free_count, 0)

            # Explicitly drop the allocator before releasing the tensor. The
            # NPUCachingAllocator's PrivatePool instance should keep the allocator
            # live until all tensors die.
            # N.B.: Deleting the pool doesn't actually do anything, since it doesn't
            # own the allocator object. But we do it anyway because it is the
            # situation that users are likely to face.
            del pool
            del allocator

            # Tensor still alive, free must NOT have been called yet.
            self.assertEqual(pa.module.get_free_count(), 0)

            del data
            torch_npu.npu.synchronize()
            torch_npu.npu.empty_cache()

            # Now that the tensor is gone, the custom allocator's free path must
            # have run for every earlier alloc.
            alloc_count = pa.module.get_alloc_count()
            free_count = pa.module.get_free_count()
            self.assertGreater(free_count, 0)
            self.assertEqual(alloc_count, free_count)
        finally:
            pa.module.reset_alloc_free_count()


instantiate_parametrized_tests(TestPluggableAllocator)

if __name__ == '__main__':
    run_tests()
