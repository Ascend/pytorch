import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
from torch_npu._inductor.config import inductor_indirect_memory_mode
from torch_npu._inductor.npu_triton_heuristics import do_bench_using_profiling_npu

IndexPutParamInfo = [
    [(10,), (5,)],
    [(1024 * 1024,), (512,)],
    [(10, 8), (5, 6)],
    
    [(30001, 128), (1024,)],
    
    [(16, 32, 64), (8, 16)],
    [(8, 4, 16, 32), (4, 4, 1)],
    [(10, 8, 6, 4), (5, 3, 2, 2)],
    [(10, 8, 6, 4, 3), (5, 2, 2, 2, 2)],
    [(10, 8, 6, 4, 3, 2), (5, 2, 2, 2, 2, 2)],
    [(10, 8, 6, 4, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)],
    [(10, 8, 6, 4, 3, 2, 2, 2), (5, 2, 2, 2, 2, 2, 2, 2)],
    
    [(3001, 15), (199,)],
    [(127, 127), (63, 63)], 

    [(100000,), (50000,)], 
]



class TestAtenIndexPutSimt(TestUtils):
    def index_put(self, table, indices, values, accumulate=False):
        return torch.index_put(table, indices, values, accumulate=accumulate)

    @parametrize('shape_info', IndexPutParamInfo)
    def test_aten_index_put(self, shape_info, enable_profiling=False):
        table_shape, index_shape = shape_info
        total_indices = torch.Size(index_shape).numel()
        total_elements = 1
        for s in table_shape:
            total_elements *= s
        linear_indices = torch.randperm(total_elements)[:total_indices]
        multi_indices = []
        temp = linear_indices.clone()
        for dim_size in reversed(table_shape):
            multi_indices.append((temp % dim_size).to(torch.int32))
            temp = temp // dim_size
        multi_indices.reverse()
        indices = tuple(idx.reshape(index_shape).npu() for idx in multi_indices)
        values = torch.randn(index_shape, dtype=torch.float32, device='npu')
        table_eager = torch.randn(table_shape, dtype=torch.float32, device='npu')
        table_compiled = table_eager.clone()
        index_put_triton = torch.compile(self.index_put, backend="inductor", dynamic=False)

        def eager_fn():
            return self.index_put(table_eager.clone(), indices, values, accumulate=False)
        
        def inductor_fn():
            return index_put_triton(table_compiled.clone(), indices, values, accumulate=False)

        r = eager_fn()
        r1 = inductor_fn()
        self.assertEqual(r, r1)

        if enable_profiling:
            eager_time = do_bench_using_profiling_npu(eager_fn)
            inductor_time = do_bench_using_profiling_npu(inductor_fn)
            print(f"Scatter benchmark, input_shape: {table_shape}, indices shape: {index_shape}, eager: {eager_time}, inductor_time: {inductor_time}")

instantiate_parametrized_tests(TestAtenIndexPutSimt)

if __name__ == "__main__":
    if inductor_indirect_memory_mode:
        run_tests()
