import contextlib
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
    _saved_fraction = None

    @classmethod
    def setUpClass(cls):
        cls._saved_allocator_settings = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
        cls._saved_fraction = torch_npu.npu.get_per_process_memory_fraction()
        torch.npu.memory._set_allocator_settings("expandable_segments:False")

    @classmethod
    def tearDownClass(cls):
        torch_npu.npu.set_per_process_memory_fraction(cls._saved_fraction)
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

    @serialTest()
    def test_mempool_with_allocator(self):
        # Exercises use_count transitions and pool.snapshot() segment counts
        # as the pool accumulates allocations.
        # NPU large segment size is 20 MiB (kLargeBuffer), so each allocation
        # between 1 MiB (kSmallSize) and 10 MiB (kMinLargeAlloc) gets a 20 MiB
        # block. Use 7 MiB per tensor: 2 fit in one 20 MiB segment, but 3 force
        # a second segment.
        pa = get_pluggable_allocator()
        if pa.module is None:
            self.skipTest("pluggable allocator module not available")
        allocator = torch_npu.npu.memory.NPUPluggableAllocator(
            os.path.join(pa.build_directory, 'pluggable_allocator_extensions.so'),
            'my_malloc', 'my_free')
        pool = torch_npu.npu.MemPool(allocator._allocator)

        # pool's use count should be 1 at this point as MemPool object
        # holds a reference
        self.assertEqual(pool.use_count(), 1)

        pa.module.reset_alloc_free_count()
        try:
            # 7 MiB of float32 elements
            nelem_7mb = 7 * 1024 * 1024 // 4

            with torch_npu.npu.use_mem_pool(pool):
                # torch.empty avoids RNG state allocation that would create an
                # extra small-pool segment on NPU (unlike CUDA where torch.randn
                # does not trigger an internal side-allocation).
                out_0 = torch.empty(nelem_7mb, device="npu")

                # use_mem_pool takes a reference, bumping use_count to 2
                self.assertEqual(pool.use_count(), 2)

            # context exit drops the reference, use_count back to 1
            self.assertEqual(pool.use_count(), 1)

            # the custom allocator's malloc must have been called for out_0
            self.assertGreater(pa.module.get_alloc_count(), 0)
            self.assertEqual(pa.module.get_free_count(), 0)

            with torch_npu.npu.use_mem_pool(pool):
                # First 7 MiB allocation — one 20 MiB segment, ~13 MiB free.
                self.assertEqual(len(pool.snapshot()), 1)

                out_1 = torch.empty(nelem_7mb, device="npu")
                # Second 7 MiB fits in the remaining ~13 MiB, still one segment.
                self.assertEqual(len(pool.snapshot()), 1)

                out_2 = torch.empty(nelem_7mb, device="npu")
                # Third 7 MiB cannot fit (~6 MiB free < 7 MiB), forces a second
                # 20 MiB segment.
                self.assertEqual(len(pool.snapshot()), 2)

            self.assertEqual(len(pool.snapshot()), 2)

            del out_0, out_1, out_2
            del pool

            # pool's destructor calls emptyCache(), which must drive the custom
            # allocator's free path.
            torch_npu.npu.synchronize()
            torch_npu.npu.empty_cache()
            self.assertGreater(pa.module.get_free_count(), 0)
        finally:
            pa.module.reset_alloc_free_count()

    def test_mempool_empty_cache(self):
        # Aligned with PyTorch's test_mempool_empty_cache: pool cache must
        # outlive the pool object so subsequent empty_cache() can reclaim it.
        torch_npu.npu.empty_cache()
        pool = torch_npu.npu.MemPool()
        x = torch.empty(1024, 1024, device="npu")

        with torch_npu.npu.use_mem_pool(pool):
            y = torch.empty(1024, 1024, device="npu")

        del y
        del x
        del pool
        segments = torch_npu.npu.memory._snapshot()["segments"]
        self.assertGreater(len(segments), 0, "expected at least one cached segment")

    def test_pool_id_in_snapshot(self):
        # Aligned with PyTorch's test_pool_id_in_snapshot: the snapshot must
        # tag each segment and each trace entry with the owning pool's id.
        try:
            torch_npu.npu.empty_cache()
            torch_npu.npu.memory._record_memory_history("all")

            pool = torch_npu.npu.MemPool()
            with torch_npu.npu.use_mem_pool(pool):
                x = torch.rand(64, device="npu")

            ss = torch_npu.npu.memory._snapshot()

            # segment_pool_id should match the MemPool id
            found_segment = False
            for seg in ss["segments"]:
                if seg["segment_pool_id"] == pool.id:
                    found_segment = True
                    break
            self.assertTrue(found_segment, "no segment tagged with pool.id")

            # trace entries for this allocation should carry pool_id
            found_trace = False
            for trace in ss["device_traces"]:
                for te in trace:
                    if te.get("pool_id") == pool.id and te["action"] == "alloc":
                        found_trace = True
                        break
            self.assertTrue(found_trace, "no alloc trace tagged with pool.id")

            del x
        finally:
            torch_npu.npu.memory._record_memory_history(None)

    def test_nested_mempool(self):
        # Aligned with PyTorch's test_nested_mempool: nested use_mem_pool
        # contexts must route allocations to the innermost pool, and each pool
        # keeps the segments allocated inside it.
        torch_npu.npu.empty_cache()
        pool1 = torch_npu.npu.MemPool()
        pool2 = torch_npu.npu.MemPool()
        pool3 = torch_npu.npu.MemPool()

        data = []
        nelem_1mb = 1024 * 1024 // 4

        def allocate_data():
            data.append(torch.empty(nelem_1mb * 20, device="npu"))

        with torch_npu.npu.use_mem_pool(pool1):
            allocate_data()
            with torch_npu.npu.use_mem_pool(pool2):
                allocate_data()
                with torch_npu.npu.use_mem_pool(pool3):
                    allocate_data()
                allocate_data()
            allocate_data()

        # Each pool keeps the segments allocated inside it. With 20 MB
        # allocations (large pool), each allocation lands in its own segment:
        # pool1 = 2, pool2 = 2, pool3 = 1.
        s1 = torch_npu.npu.memory.memory_snapshot(pool1.id)
        s2 = torch_npu.npu.memory.memory_snapshot(pool2.id)
        s3 = torch_npu.npu.memory.memory_snapshot(pool3.id)
        self.assertEqual(len(s1), 2)
        self.assertEqual(len(s2), 2)
        self.assertEqual(len(s3), 1)

    def test_mempool_emptycache_multithread(self):
        # Aligned with PyTorch's test_mempool_emptycache_multithread:
        # empty_cache() inside a use_mem_pool context must NOT release the
        # pool's cached segments.
        num_threads = 4

        def my_function(pool):
            with torch_npu.npu.use_mem_pool(pool):
                x = torch.randn(4, device="npu")
                del x
                torch_npu.npu.empty_cache()

        pools = [torch_npu.npu.MemPool() for _ in range(num_threads)]
        threads = [
            threading.Thread(target=my_function, args=(pools[i],))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # empty_cache should have done nothing under mempool context: each
        # pool still holds its single segment.
        for p in pools:
            self.assertEqual(len(p.snapshot()), 1, "expected a single cached segment")

    def _setup_mempool_limited_memory(self, additional_allowed_mb):
        # Limit memory for mempool tests.
        torch_npu.npu.empty_cache()
        mb = 1024 * 1024
        _, all_memory = torch_npu.npu.mem_get_info()
        pre_reserved = torch_npu.npu.memory_reserved()
        total_allowed = additional_allowed_mb * mb + pre_reserved
        fraction = total_allowed / all_memory
        torch_npu.npu.set_per_process_memory_fraction(fraction)
        return torch_npu.npu.current_device(), torch.int8

    def _teardown_mempool_limited_memory(self):
        torch_npu.npu.empty_cache()
        torch_npu.npu.set_per_process_memory_fraction(self._saved_fraction)

    def _alloc_record_free_sync(self, pool_ctx, s2, nbytes):
        # Helper for block_free_not_deferred test.
        with pool_ctx:
            a = torch.empty(nbytes, dtype=torch.uint8, device="npu")
            original_ptr = a.data_ptr()

            with torch_npu.npu.stream(s2):
                a.record_stream(s2)
                _ = a + 1

            del a
            torch_npu.npu.synchronize()

            b = torch.empty(nbytes, dtype=torch.uint8, device="npu")
            new_ptr = b.data_ptr()
            del b
        return original_ptr, new_ptr

    @serialTest()
    def test_mempool_release_cached_blocks_during_diversion(self):
        # Regression: release_cached_blocks during mempool diversion.
        # `torch_npu.npu.use_mem_pool(...)` would skip the OOM-time
        # release_cached_blocks() retry, because the gate read
        # `captures_underway.empty()` — which is non-empty during any
        # private-pool diversion, even when no real stream capture
        # is active. The fix uses a separate `num_active_captures_` counter
        # tracked by NPUGraph::capture_begin/end, so default-pool cached
        # blocks can still be reclaimed during plain mempool diversion.
        nelem_1mb = 1024 * 1024  # int8 elements per MiB
        device, dtype = self._setup_mempool_limited_memory(80)
        try:
            # Reserve 60 MB on the default pool, then free -> cached but not
            # returned to the driver. memory_reserved stays at 60 MB.
            a = torch.empty(60 * nelem_1mb, device=device, dtype=dtype)
            del a

            pool = torch_npu.npu.MemPool()
            with torch_npu.npu.use_mem_pool(pool):
                # 60 MB into the private pool. Reservation budget remaining
                # is only 20 MB (80 cap - 60 cached), so aclrtMalloc must
                # fail. The OOM retry needs to call release_cached_blocks()
                # on the default pool to free the 60 MB and retry. Pre-fix,
                # that path was gated off and this would raise.
                b = torch.empty(60 * nelem_1mb, device=device, dtype=dtype)
            self.assertEqual(b.numel(), 60 * nelem_1mb)
            del b
            del pool
        finally:
            self._teardown_mempool_limited_memory()

    def test_mempool_oom_recovery_releases_cached_blocks(self):
        # OOM recovery releases cached blocks with user mempool active.
        MB = 1024 * 1024

        def align_down_2mb(n):
            return n & ~(2 * MB - 1)

        for label, make_ctx in [
            ("default pool", lambda: contextlib.nullcontext()),
            ("user mempool",
             lambda: torch_npu.npu.use_mem_pool(torch_npu.npu.MemPool())),
        ]:
            with self.subTest(label=label):
                torch_npu.npu.empty_cache()
                free_before = torch_npu.npu.mem_get_info()[0]

                fill_size = align_down_2mb(free_before // 2)
                if fill_size < 64 * MB:
                    self.skipTest("Not enough NPU memory for this test")

                filler = torch.empty(fill_size, dtype=torch.uint8, device="npu")
                del filler

                alloc_size = align_down_2mb(free_before - free_before // 8)
                oom = False
                try:
                    with make_ctx():
                        big = torch.empty(alloc_size, dtype=torch.uint8, device="npu")
                        del big
                except torch.OutOfMemoryError:
                    oom = True

                self.assertFalse(
                    oom,
                    f"[{label}] OOM even though the default pool had "
                    f"{fill_size // MB} MiB of freeable cached blocks "
                    "-- release_cached_blocks was likely skipped",
                )

    def test_mempool_block_free_not_deferred(self):
        # Multi-stream block free NOT deferred during user mempool.
        torch_npu.npu.empty_cache()
        s2 = torch_npu.npu.Stream()
        nbytes = 1024 * 1024

        for label, pool_ctx in [
            ("default pool", contextlib.nullcontext()),
            ("user mempool",
             torch_npu.npu.use_mem_pool(torch_npu.npu.MemPool())),
        ]:
            with self.subTest(label=label):
                torch_npu.npu.empty_cache()
                orig, new = self._alloc_record_free_sync(pool_ctx, s2, nbytes)
                self.assertEqual(
                    new,
                    orig,
                    lambda msg: f"{msg}\n[{label}] Block not reused after multi-stream free "
                    "-- free was likely deferred as if under graph capture",
                )


class TestMemPoolUseOnOOM(TestCase):
    """Behavioral tests for MemPool use_on_oom.

    Aligned with PyTorch's test_mempool_limited_memory_with_allocator and
    test_deleted_mempool_not_used_on_oom: a use_on_oom pool must serve
    allocations when the default pool OOMs, and a deleted use_on_oom pool
    must NOT be reused.
    """

    _saved_allocator_settings = None
    _saved_fraction = None

    @classmethod
    def setUpClass(cls):
        cls._saved_allocator_settings = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
        cls._saved_fraction = torch_npu.npu.get_per_process_memory_fraction()
        torch.npu.memory._set_allocator_settings("expandable_segments:False")

    @classmethod
    def tearDownClass(cls):
        torch_npu.npu.set_per_process_memory_fraction(cls._saved_fraction)
        torch.npu.memory._set_allocator_settings(cls._saved_allocator_settings)

    def _setup_limited_memory(self, additional_allowed_mb):
        # Limit the caching allocator to (already_reserved + additional_allowed_mb).
        torch_npu.npu.empty_cache()
        mb = 1024 * 1024
        _, all_memory = torch_npu.npu.mem_get_info()
        pre_reserved = torch_npu.npu.memory_reserved()
        total_allowed = additional_allowed_mb * mb + pre_reserved
        fraction = total_allowed / all_memory
        torch_npu.npu.set_per_process_memory_fraction(fraction)
        return torch_npu.npu.current_device(), torch.int8

    def _teardown_limited_memory(self):
        torch_npu.npu.empty_cache()
        torch_npu.npu.set_per_process_memory_fraction(self._saved_fraction)

    @serialTest()
    def test_mempool_limited_memory_with_allocator(self):
        # use_on_oom=True pool must absorb allocations when the default pool
        # OOMs; use_on_oom=False pool must NOT.
        # NPU: large allocations round up to the next 2 MiB boundary
        # (kRoundLarge), so a 40 MiB tensor may need up to 42 MiB. The budget
        # adds headroom (88 MiB) to account for this rounding overhead.
        # torch.empty is used instead of torch.randn to avoid extra RNG state
        # allocations in the default pool on NPU.
        pool_do_not_use = torch_npu.npu.MemPool()
        pool_use = torch_npu.npu.MemPool(use_on_oom=True)

        nelem_1mb = 1024 * 1024 // 4
        self._setup_limited_memory(88)
        try:
            # remaining free mem: 88 mb
            # mempool_use [] 0 mb
            # mempool_do_not_use [] 0 mb
            # default pool [] 0 mb
            with torch_npu.npu.use_mem_pool(pool_do_not_use):
                a = torch.empty(40 * nelem_1mb, device="npu")
            with torch_npu.npu.use_mem_pool(pool_use):
                b = torch.empty(40 * nelem_1mb, device="npu")
            a_dataptr = a.data_ptr()
            b_dataptr = b.data_ptr()
            # remaining free mem: 8 mb
            # mempool_do_not_use [aaaa] 40 mb
            # mempool_use [bbbb] 40 mb
            # default pool [] 0 mb

            with self.assertRaises(torch.OutOfMemoryError):
                # out of memory
                c = torch.empty(40 * nelem_1mb, device="npu")

            del a, b
            # remaining free mem: 8 mb
            # mempool_do_not_use [____] 40 mb
            # mempool_use [____] 40 mb
            # default pool [] 0 mb

            # c should not oom and instead can use mempool_use as fallback
            c = torch.empty(30 * nelem_1mb, device="npu")
            c_dataptr = c.data_ptr()
            # remaining free mem: 8 mb
            # mempool_do_not_use [____] 40 mb
            # mempool_use [ccc_] 40 mb
            # default pool [] 0 mb

            # pool_do_not_use must not be used as fallback.
            with self.assertRaises(torch.OutOfMemoryError):
                # out of memory since can't use mempool_do_not_use
                d = torch.empty(30 * nelem_1mb, device="npu")

            del c
            # remaining free mem: 8 mb
            # mempool_do_not_use [____] 40 mb
            # mempool_use [____] 40 mb
            # default pool [] 0 mb

            # c reused b's address because b's pool served the fallback.
            self.assertEqual(b_dataptr, c_dataptr)

            # make sure we can still use mempool_use as intended after c is deleted
            with torch_npu.npu.use_mem_pool(pool_use):
                e = torch.empty(20 * nelem_1mb, device="npu")
            # remaining free mem: 8 mb
            # mempool_do_not_use [____] 40 mb
            # mempool_use [ee__] 40 mb
            # default pool [] 0 mb

            e_dataptr = e.data_ptr()
            del e
            self.assertEqual(e_dataptr, c_dataptr)

            # pool's destructor calls emptyCache()
            del pool_use, pool_do_not_use
        finally:
            self._teardown_limited_memory()

    @serialTest()
    def test_deleted_mempool_not_used_on_oom(self):
        # Regression for ~MemPool() calling setUseOnOOM(id, false): a deleted
        # use_on_oom pool must be removed from use_on_oom_pools and never
        # serve fallback allocations.
        # NPU: large allocations round up to the next 2 MiB boundary
        # (kRoundLarge), so a 40 MiB tensor may need up to 42 MiB. The budget
        # adds headroom (48 MiB) to account for this rounding overhead.
        # torch.empty is used instead of torch.randn to avoid extra RNG state
        # allocations in the default pool on NPU.
        nelem_1mb = 1024 * 1024 // 4

        self._setup_limited_memory(48)
        try:
            # Create + delete many use_on_oom pools; each one's dtor must
            # unregister it from use_on_oom_pools.
            for _ in range(10):
                pool_use_on_oom = torch_npu.npu.MemPool(use_on_oom=True)
                with torch_npu.npu.use_mem_pool(pool_use_on_oom):
                    a = torch.empty(40 * nelem_1mb, device="npu")
                del a
                del pool_use_on_oom

            # One live use_on_oom pool holding all 40 MB.
            new_pool = torch_npu.npu.MemPool(use_on_oom=True)
            with torch_npu.npu.use_mem_pool(new_pool):
                a = torch.empty(40 * nelem_1mb, device="npu")
            del a

            # Fallback must succeed (uses new_pool) and not crash on the
            # deleted pools.
            b = torch.empty(20 * nelem_1mb, device="npu")
            c = torch.empty(20 * nelem_1mb, device="npu")

            del b, c, new_pool
        finally:
            self._teardown_limited_memory()


class TestMemPoolNoSplit(TestCase):
    """Behavioral tests for MemPool no_split.

    Aligned with PyTorch's test_mempool_no_split: a no_split pool must not
    split segments, so each segment holds exactly one block and the no_split
    pool has more segments (and fewer blocks) than the split pool.
    """

    @serialTest()
    def test_mempool_no_split(self):
        torch_npu.npu.empty_cache()

        pool_split = torch_npu.npu.MemPool()  # default: no_split=False
        pool_no_split = torch_npu.npu.MemPool(no_split=True)

        # 1 MB in number of float32 elements.
        nelem_1mb = 1024 * 1024 // 4

        # First 4 MB allocation in each pool: both get one segment.
        with torch_npu.npu.use_mem_pool(pool_split):
            a_split = torch.randn(4 * nelem_1mb, device="npu")
        with torch_npu.npu.use_mem_pool(pool_no_split):
            a_no_split = torch.randn(4 * nelem_1mb, device="npu")

        # Second 4 MB allocation: split pool reuses the leftover of the
        # existing segment; no_split pool must open a new segment.
        with torch_npu.npu.use_mem_pool(pool_split):
            b_split = torch.randn(4 * nelem_1mb, device="npu")
        with torch_npu.npu.use_mem_pool(pool_no_split):
            b_no_split = torch.randn(4 * nelem_1mb, device="npu")

        snap_split = pool_split.snapshot()
        snap_no_split = pool_no_split.snapshot()

        # no_split pool should have strictly more segments.
        if len(snap_no_split) <= len(snap_split):
            raise AssertionError(
                f"Expected no_split pool to have more segments, "
                f"but got {len(snap_no_split)} vs {len(snap_split)}")

        # Every no_split segment holds exactly one block (no splitting).
        for seg in snap_no_split:
            if len(seg["blocks"]) != 1:
                raise AssertionError(
                    f"Expected 1 block in no_split segment, got {len(seg['blocks'])}")

        def count_blocks(snaps):
            return sum(len(seg["blocks"]) for seg in snaps)

        blocks_split = count_blocks(snap_split)
        blocks_no_split = count_blocks(snap_no_split)

        # no_split pool has fewer blocks because it doesn't keep a leftover
        # block for the remaining memory in each segment.
        self.assertLess(
            blocks_no_split, blocks_split,
            f"Expected no_split pool to have fewer blocks, "
            f"but got {blocks_no_split} vs {blocks_split}")

        del a_split, b_split, a_no_split, b_no_split
        del pool_split, pool_no_split


instantiate_parametrized_tests(TestPluggableAllocator)

if __name__ == '__main__':
    run_tests()
