import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
from torch_npu._inductor.config import inductor_indirect_memory_mode
from torch_npu._inductor.npu_triton_heuristics import do_bench_using_profiling_npu


GatherParamInfo = [
    [(10, 5, 8), (2, 3, 4)],
    [(100, 16, 32), (50, 16, 10)],
    [(5, 5, 5), (5, 5, 5)],
    
    [(100,), (50,)],
    [(10, 5, 8, 8), (2, 3, 4, 4)],
    
    [(4, 16), (2, 128)],
    [(4, 32), (2, 128)],
    [(4, 64), (2, 128)],
    [(4, 128), (2, 128)],
    
    [(4, 3), (2, 128)],
    [(4, 33), (2, 128)],
    [(4, 129), (2, 128)],
    
    [(10001, 8), (2, 128 * 50)],
    [(10001, 16), (2, 128 * 50)],
    [(10001, 32), (2, 128 * 50)],
    [(10001, 64), (2, 128 * 50)],
    [(10001, 128), (2, 128 * 50)],
    
    [(2048, 16), (2048, 1)],
    [(1, 100, 32), (1, 50, 10)],
    [(169219, 16), (169219, 1)],
    [(10, 32000), (2, 100)],
]


class TestAtenGatherSimt(TestUtils):
    def gather(self, table, index):
        return torch.ops.aten.gather.default(table, -1, index)

    @parametrize('shape_info', GatherParamInfo)
    def test_aten_gather(self, shape_info, enable_profiling=False):
        [table_shape, index_shape] = shape_info
        table_dim = table_shape[-1]
        index = torch.randint(0, table_dim, size=index_shape, dtype=torch.int64).npu()
        table = torch.randn(table_shape, requires_grad=False, dtype=torch.float32, device='npu')

        
        gather_triton = torch.compile(self.gather, backend="inductor", dynamic=False)

        def eager_fn():
            return self.gather(table, index)
        
        def inductor_fn():
            return gather_triton(table, index)

        r = eager_fn()
        r1 = inductor_fn()
        self.assertEqual(r, r1)

        if enable_profiling:
            eager_time = do_bench_using_profiling_npu(eager_fn)
            inductor_time = do_bench_using_profiling_npu(inductor_fn)
            print(f"Gather benchmark, table_shape: {table_shape}, index shape: {index_shape}, eager: {eager_time}, inductor_time: {inductor_time}")

instantiate_parametrized_tests(TestAtenGatherSimt)

if __name__ == "__main__":
    if inductor_indirect_memory_mode:
        run_tests()
