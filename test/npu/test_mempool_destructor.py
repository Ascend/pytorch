import torch
from torch_npu.testing.testcase import TestCase, run_tests


class TestMemPoolDestructor(TestCase):
    def test_empty_cache_mempool_destruction(self):
        # make sure clear other task mem
        torch.npu.synchronize()
        torch.npu.empty_cache()

        pool_memory_reserved = 0
        pool = torch.npu.MemPool()
        with torch.npu.use_mem_pool(pool):
            tensor = torch.full((1024 ** 3,), 53, dtype=torch.uint8, device='npu')
            pool_memory_reserved = torch.npu.memory_reserved()
            del tensor

        torch.npu.synchronize()
        torch.npu.empty_cache()
        pool_before_del_memory_reserved = torch.npu.memory_reserved()
        self.assertGreater(pool_memory_reserved, pool_before_del_memory_reserved / 2)
        self.assertGreater(pool_before_del_memory_reserved, pool_memory_reserved / 2)

        # should free mempool cache
        del pool
        pool_after_del_memory_reserved = torch.npu.memory_reserved()
        self.assertGreater(pool_memory_reserved / 2, pool_after_del_memory_reserved)

        # normally empty cache after mempool processing
        tensor_2 = torch.full((1024 ** 3,), 53, dtype=torch.uint8, device='npu')
        del tensor_2
        torch.npu.synchronize()
        torch.npu.empty_cache()
        end_memory_reserved = torch.npu.memory_reserved()
        self.assertGreater(pool_memory_reserved / 2, end_memory_reserved)

    def test_empty_cache_mempool_destruction_abnormal(self):
        test_result = True
        try:
            pool = torch.npu.MemPool()
            del pool
        except Exception as e:
            test_result = False
        self.assertTrue(test_result)


if __name__ == '__main__':
    run_tests()
