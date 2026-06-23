import os
import math
import torch
import torch_npu

import torch_npu.npu.utils as utils

from torch_npu.testing.testcase import TestCase, run_tests


class TestAllocator(TestCase):
    def test_huge_memory_alloc_20M(self):
        prev = torch_npu.npu.memory_allocated()
        a = torch.rand(1024 * 1024 * 40, dtype=torch.float32).npu()
        # 实际申请1G内存
        version = utils.get_cann_version(module="CANN")
        if (utils.get_soc_version() >= 260 and version >= "9.1.0"):
            self.assertEqual(torch_npu.npu.memory_allocated(), prev + math.ceil((4 * 40 * 1024 * 1024) / 512) * 512)
        else:
            self.assertEqual(torch_npu.npu.memory_allocated(),
                             prev + math.ceil((4 * 40 * 1024 * 1024 + 32) / 512) * 512)

    def test_huge_memory_alloc_512B(self):
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
        prev = torch_npu.npu.memory_allocated()
        a = torch.rand(8 * 8 * 16, dtype=torch.float32).npu()  # 512B
        # 实际申请1M内存
        version = utils.get_cann_version(module="CANN")
        if (utils.get_soc_version() >= 260 and version >= "9.1.0"):
            self.assertEqual(torch_npu.npu.memory_allocated(), prev + math.ceil((8 * 8 * 16 * 4) / 512) * 512)
        else:
            self.assertEqual(torch_npu.npu.memory_allocated(), prev + math.ceil((8 * 8 * 16 * 4 + 32) / 512) * 512)

    def test_huge_memory_alloc_512B_by_vm(self):
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
        prev = torch_npu.npu.memory_allocated()
        a = torch.rand(8 * 8 * 16, dtype=torch.float32).npu()  # 512B
        # 实际申请1M内存
        version = utils.get_cann_version(module="CANN")
        if (utils.get_soc_version() >= 260 and version >= "9.1.0"):
            self.assertEqual(torch_npu.npu.memory_allocated(), prev + math.ceil((8 * 8 * 16 * 4) / 512) * 512)
        else:
            self.assertEqual(torch_npu.npu.memory_allocated(), prev + math.ceil((8 * 8 * 16 * 4 + 32) / 512) * 512)
        del os.environ["PYTORCH_NPU_ALLOC_CONF"]

    def test_max_non_split_rounding_mb(self):
        device = torch.device("npu")

        def get_reserved_memory():
            return torch.npu.memory_stats(device)["reserved_bytes.all.current"]

        def free_large_and_allocate_small(confg_str):
            # key: set max_split_size_mb and max_non_split_rounding_mb
            # testcase: set max_split_size_mb=50MB, max_non_split_rounding_mb=xxxMB
            torch.npu.memory._set_allocator_settings(confg_str)
            torch.npu.memory.empty_cache()
            torch.npu.synchronize()
            # 100M Tensor and 75M Tensor
            large_size = 100 * 1024 * 1024 // 4
            small_size = 75 * 1024 * 1024 // 4
            # 1. allocate 100MB(>50MB), then free
            large_tensor = torch.randn(large_size, device=device)
            torch.npu.synchronize()
            # free, but memory reserved
            del large_tensor
            torch.npu.synchronize()
            # 2. allocate 75MB(>50MB), 100-75=25MB > 20MB -> need align
            before_alloc = get_reserved_memory()
            small_tensor = torch.randn(small_size, device=device)
            torch.npu.synchronize()
            after_alloc = get_reserved_memory()
            allocated_mem = after_alloc - before_alloc
            # clean
            del small_tensor
            torch.npu.synchronize()
            torch.npu.memory.empty_cache()
            torch.npu.synchronize()
            return allocated_mem

        allocated_mem = free_large_and_allocate_small("max_split_size_mb:50,max_non_split_rounding_mb:20")
        # Expected value is equal 75M, but may have small fluctuation
        self.assertTrue(allocated_mem >= 70 * 1024 * 1024)

        allocated_mem = free_large_and_allocate_small("max_split_size_mb:50,max_non_split_rounding_mb:30")
        # Expected value is equal zero, but may have small fluctuation
        self.assertTrue(allocated_mem < 10 * 1024 * 1024)

    def test_release_lock_on_npumalloc(self):
        # Real testing is performance testing, which need dedicated machine
        # release_lock_on_npumalloc True should have equal or better performance than False
        # Here just function testing/ut
        set_config = True
        try:
            torch.npu.memory._set_allocator_settings("release_lock_on_npumalloc:True")
        except Exception as e:
            set_config = False
        self.assertTrue(set_config)


if __name__ == '__main__':
    run_tests()
