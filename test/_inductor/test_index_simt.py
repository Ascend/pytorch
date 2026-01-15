import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
from torch_npu._inductor.config import arch_support_simt
from torch_npu._inductor.npu_triton_heuristics import do_bench_using_profiling_npu

# table_shape, index_shape
IndexParamInfo = [
    [(10001, 128), (1, 128 * 50)],
    [(2048, 256), (32, 128)],
    [(101, 67), (13, 19)],
    [(512, 4096), (16, 512)],
    [(100000, 64), (128, 64)],
    [(5000, 128), (8, 4, 32)],
    [(100, 100), (1, 1)],
    [(8192, 512), (512, 256)],
]


class TestAtenIndexSimt(TestUtils):
    def index(self, table, index):
        return torch.ops.aten.index(table, index)

    @parametrize('shape_info', IndexParamInfo)
    def test_aten_index(self, shape_info, enable_profiling=False):
        [table_shape, index_shape] = shape_info
        embedding_vocabulary = table_shape[0]
        embedding_dim = table_shape[1]
        indice_x = torch.randint(0, embedding_vocabulary, size=index_shape, dtype=torch.int32).npu()
        indice_y = torch.randint(0, embedding_dim, size=index_shape, dtype=torch.int32).npu()
        index = (indice_x, indice_y)
        table = torch.randn(table_shape, requires_grad=False, dtype=torch.float32, device='npu')
        
        index_triton = torch.compile(self.index, backend="inductor", dynamic=False)

        def eager_fn():
            return self.index(table, index)
        
        def inductor_fn():
            return index_triton(table, index)

        r = eager_fn()
        r1 = inductor_fn()
        self.assertEqual(r, r1)

        if enable_profiling:
            eager_time = do_bench_using_profiling_npu(eager_fn)
            inductor_time = do_bench_using_profiling_npu(inductor_fn)
            print(f"Index benchmark, table_shape: {table_shape}, index shape: {index_shape}, eager: {eager_time}, inductor_time: {inductor_time}")

instantiate_parametrized_tests(TestAtenIndexSimt)

if __name__ == "__main__":
    if arch_support_simt:
        run_tests()
