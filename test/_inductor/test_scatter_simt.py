import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
from torch_npu._inductor.config import inductor_indirect_memory_mode
from torch_npu._inductor.npu_triton_heuristics import do_bench_using_profiling_npu

ScatterParamInfo = [
    [(128,), (128,), (128,)],
    [(32, 256), (32, 256), (32, 256)],
    [(33, 53), (33, 53), (33, 53)],
    [(4096, 8), (4096, 8), (4096, 8)],
    [(2, 131072), (2, 131072), (2, 131072)],
    [(10, 100), (10, 5), (10, 5)], 
    [(10, 100), (10, 100), (10, 100)],
    [(2, 2, 2, 4, 16), (2, 2, 2, 4, 16), (2, 2, 2, 4, 16)],
]


def generate_no_collision_index(input_shape, index_shape, dim=-1):
    input_dim_size = input_shape[dim]
    rows = torch.Size(index_shape[:-1]).numel() if len(index_shape) > 1 else 1
    
    all_rows = []
    for _ in range(rows):
        row_idx = torch.randperm(input_dim_size)[:index_shape[dim]]
        all_rows.append(row_idx)
    
    index = torch.stack(all_rows).reshape(index_shape)
    return index.npu()


class TestAtenScatterSimt(TestUtils):
    def scatter(self, input_tensor, index, src):
        return torch.ops.aten.scatter(input_tensor, -1, index, src)

    @parametrize('shape_info', ScatterParamInfo)
    def test_aten_scatter(self, shape_info, enable_profiling=False):
        [input_shape, index_shape, src_shape] = shape_info
        input_dim = input_shape[-1]
        index = generate_no_collision_index(input_shape, index_shape).npu()
        input_tensor = torch.randn(input_shape, requires_grad=False, dtype=torch.float32, device='npu')
        src_tensor = torch.randn(src_shape, dtype=torch.float32, device='npu')

        scatter_triton = torch.compile(self.scatter, backend="inductor", dynamic=False)

        def eager_fn():
            return scatter(input_tensor, index, src_tensor)
        
        def inductor_fn():
            return scatter_triton(input_tensor, index, src_tensor)
        
        r = eager_fn()
        r1 = inductor_fn()
        self.assertEqual(r, r1)

        if enable_profiling:
            eager_time = do_bench_using_profiling_npu(eager_fn)
            inductor_time = do_bench_using_profiling_npu(inductor_fn)
            print(f"Scatter benchmark, input_shape: {input_shape}, index shape: {index_shape}, src_shape: {src_shape}, eager: {eager_time}, inductor_time: {inductor_time}")

instantiate_parametrized_tests(TestAtenScatterSimt)

if __name__ == "__main__":
    if inductor_indirect_memory_mode:
        run_tests()
