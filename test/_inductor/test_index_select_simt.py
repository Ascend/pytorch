import gc
import unittest
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import BenchmarkTestUtils
from torch_npu._inductor.config import inductor_indirect_memory_mode
from torch_npu._inductor.npu_triton_heuristics import do_bench_using_profiling_npu


IndexSelectParamInfo = [
    # 1D
    [(10001, ), (128, ), (0, ), torch.float32, torch.int64, False],
    # 2D
    [(10001, 128), (128, ), (0, 1), torch.float32, torch.int64, False],
    # 3D
    [(10001, 128, 128), (128, ), (0, 1, 2), torch.float32, torch.int64, False],
    # 4D
    [(200, 128, 128, 64), (128, ), (0, 1, 2, 3), torch.float32, torch.int64, False],
]

IndexSelectParamInfoBenchmark = [
    # 1D
    [(10001, ), (32, ), (0, ), torch.float32, torch.int64, True],
    [(10001, ), (128, ), (0, ), torch.float32, torch.int64, True],
    [(10001, ), (1024, ), (0, ), torch.float32, torch.int64, True],

    # 2D
    [(9000, 128), (9000, ), (0, 1), torch.float32, torch.int64, True],
    [(9000, 128), (1024, ), (0, 1), torch.float32, torch.int64, True],
    [(10001, 32), (128, ), (0, 1), torch.float32, torch.int64, True],
    [(4, 128), (128, ), (0, 1), torch.float32, torch.int64, True],
    [(2048, 16), (20480, ), (0, 1), torch.float32, torch.int64, True],
    [(2048, 16), (1063, ), (0, 1), torch.float32, torch.int64, True],

    # 3D
    [(64, 32, 16), (1024, ), (0, 1, 2), torch.float32, torch.int64, True],
    [(64, 32, 16), (10000, ), (0, 1, 2), torch.float32, torch.int64, True],

    # 4D
    [(64, 32, 16, 8), (1024, ), (0, 1, 2, 3), torch.float32, torch.int64, True],
    [(64, 32, 16, 8), (10000, ), (0, 1, 2, 3), torch.float32, torch.int64, True],
]


class TestAtenIndexSelectSimt(BenchmarkTestUtils):
    profiling_file_path = "TestAtenIndexSelectSimt_pref.csv"

    def index_select(self, table, dim, indice):
        return torch.index_select(table, dim, indice)
    
    def do_single_test(self, dim, param_info):
        table_shape, index_shape, _, table_dtype, index_dtype, enable_profiling = param_info
        indice = torch.randint(0, table_shape[dim], size=index_shape, dtype=index_dtype, device='npu')
        table = torch.randn(table_shape, requires_grad=False, dtype=table_dtype, device='npu')

        index_select_triton = torch.compile(self.index_select, backend="inductor", dynamic=False)

        def eager_fn():
            return self.index_select(table, dim, indice)

        def index_select_triton_fn():
            return index_select_triton(table, dim, indice)

        r = eager_fn()
        r1 = index_select_triton_fn()
        self.assertEqual(r, r1)

        if enable_profiling:
            eager_time = do_bench_using_profiling_npu(eager_fn)
            inductor_time = do_bench_using_profiling_npu(index_select_triton_fn)
            self.write_performance_info(self.profiling_file_path, {
                "op type": "aten.index_select",
                "dim": f"{dim}",
                "shape": f"{table_shape}, {index_shape}",
                "dtype": f"{table_dtype}, {index_dtype}",
                "eager_time": f"{eager_time:.2f}",
                "inductor_time": f"{inductor_time:.2f}"
            })

    @parametrize('param_info', IndexSelectParamInfo)
    def test_aten_index_select(self, param_info):
        table_shape, index_shape, dims, table_dtype, index_dtype, enable_profiling = param_info
        for dim in dims:
            self.do_single_test(dim, param_info)

    @parametrize('param_info', IndexSelectParamInfoBenchmark)
    def test_aten_index_select_benchmark(self, param_info):
        table_shape, index_shape, dims, table_dtype, index_dtype, enable_profiling = param_info
        for dim in dims:
            self.do_single_test(dim, param_info)

    def test_index_select_unsqueeze_expand(self):

        def index_select_view(table, dim, indice):
            indice = indice.expand(1, 128)
            indice = indice.reshape(128, )
            table = table.unsqueeze(-1)
            table = table.expand(32, 64, 16)
            table = table.permute((1, 0, 2))
            return torch.index_select(table, dim, indice)

        table_shape = (64, 32, 16)
        index_shape = (128, )
        table_dtype, index_dtype = torch.float32, torch.int64
        dim = 1
        indice = torch.randint(0, table_shape[dim], size=(1, ), dtype=index_dtype, device='npu')
        table = torch.randn((32, 64), requires_grad=False, dtype=table_dtype, device='npu')
        index_select_triton = torch.compile(index_select_view, backend="inductor", dynamic=False)

        r = index_select_view(table, dim, indice)
        r1 = index_select_triton(table, dim, indice)
        self.assertEqual(r, r1)

    def test_index_select_unsqueeze_expand_pointwise(self):

        def index_select_view(table, dim, indice, table2):
            indice = indice.expand(1, 128)
            indice = indice.reshape(128, )
            table = table.unsqueeze(-1)
            table = table.expand(32, 64, 16)
            table = table.permute((1, 0, 2))
            table = table2 + table
            return torch.index_select(table, dim, indice)

        table_shape = (64, 32, 16)
        index_shape = (128, )
        table_dtype, index_dtype = torch.float32, torch.int64
        dim = 1
        indice = torch.randint(0, table_shape[dim], size=(1, ), dtype=index_dtype, device='npu')
        table = torch.randn((32, 64), requires_grad=False, dtype=table_dtype, device='npu')
        table2 = torch.randn(table_shape, requires_grad=False, dtype=table_dtype, device='npu')
        index_select_triton = torch.compile(index_select_view, backend="inductor", dynamic=False)

        r = index_select_view(table, dim, indice, table2)
        r1 = index_select_triton(table, dim, indice, table2)
        self.assertEqual(r, r1)

instantiate_parametrized_tests(TestAtenIndexSelectSimt)

if __name__ == "__main__":
    if inductor_indirect_memory_mode:
        run_tests()