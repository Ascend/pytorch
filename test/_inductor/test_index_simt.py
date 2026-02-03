import unittest
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import BenchmarkTestUtils
from torch_npu._inductor.config import inductor_indirect_memory_mode
from torch_npu._inductor.npu_triton_heuristics import do_bench_using_profiling_npu


# table_shape, index_shape
IndexParamInfo = [
    # 1D
    [(10001, ), (128, ), torch.float32, torch.int64, False],
    [(2048, ), (32, ), torch.float32, torch.int64, False],
    # 2D
    [(101, 67), (13, ), torch.float32, torch.int64, False],
    [(512, 4096), (16, ), torch.float32, torch.int64, False],
    # 3D
    [(100000, 64, 32), (128, ), torch.float32, torch.int64, False],
    [(5000, 128, 10), (8, ), torch.float32, torch.int64, False],
    # 4D
    [(100, 100, 100, 100), (8, ), torch.float32, torch.int64, False],
]

IndexSelectParamInfoBenchmark = [
    # 1D
    [(2048, ), (32, ), torch.float32, torch.int64, True],
    [(2048, ), (64, ), torch.float32, torch.int64, True],
    [(2048, ), (128, ), torch.float32, torch.int64, True],
    [(2048, ), (256, ), torch.float32, torch.int64, True],
    [(2048, ), (2048, ), torch.float32, torch.int64, True],

    # 2D
    [(128, 2), (100, ), torch.float32, torch.int64, True],
    [(2, 1024), (100, ), torch.float32, torch.int64, True],
    [(2, 1024), (16, ), torch.float32, torch.int64, True],
    [(5000, 64), (16, ), torch.float32, torch.int64, True],
    [(5000, 64), (16, ), torch.float32, torch.int64, True],

    # 3D
    [(128, 2, 2), (128, ), torch.float32, torch.int64, True],
    [(2, 128, 2), (128, ), torch.float32, torch.int64, True],
    [(2, 2, 1024), (128, ), torch.float32, torch.int64, True],
    [(2, 1, 1024), (128, ), torch.float32, torch.int64, True],
    [(2, 1, 1024), (64, ), torch.float32, torch.int64, True],

    # 4D
    [(128, 2, 2, 2), (128, ), torch.float32, torch.int64, True],
    [(2, 128, 2, 2), (128, ), torch.float32, torch.int64, True],
    [(2, 2, 128, 2), (128, ), torch.float32, torch.int64, True],
    [(16, 3, 244, 244), (4, ), torch.float32, torch.int64, True],
    [(64, 128, 14, 14), (64, ), torch.float32, torch.int64, True],
]


class TestAtenIndexSimt(BenchmarkTestUtils):
    profiling_file_path = "TestAtenIndexSimt_perf.csv"

    def index(self, table, index):
        return torch.ops.aten.index(table, index)
    
    def do_test_aten_index_single(self, para_info):
        [table_shape, indice_shape, table_dtype, indice_dtype, enable_profiling] = para_info
        indices = []
        for bound in table_shape:
            indice = torch.randint(0, bound, size=indice_shape, dtype=indice_dtype, device='npu')
            indices.append(indice)
        table = torch.randn(table_shape, requires_grad=False, dtype=table_dtype, device='npu')

        index_triton = torch.compile(self.index, backend="inductor", dynamic=False)

        def eager_fn():
            return self.index(table, indices)
        
        def inductor_fn():
            return index_triton(table, indices)

        r = eager_fn()
        r1 = inductor_fn()
        self.assertEqual(r, r1)

        if enable_profiling:
            eager_time = do_bench_using_profiling_npu(eager_fn)
            inductor_time = do_bench_using_profiling_npu(inductor_fn)
            self.write_performance_info(self.profiling_file_path, {
                "op type": "aten.index",
                "shape": f"{table_shape}, {indice_shape}",
                "dtype": f"{table_dtype}, {indice_dtype}",
                "eager_time": f"{eager_time:.2f}",
                "inductor_time": f"{inductor_time:.2f}"
            })

    @parametrize('para_info', IndexParamInfo)
    def test_aten_index(self, para_info):
        self.do_test_aten_index_single(para_info)

    @parametrize('para_info', IndexSelectParamInfoBenchmark)
    def test_aten_index_benchmark(self, para_info):
        self.do_test_aten_index_single(para_info)

instantiate_parametrized_tests(TestAtenIndexSimt)

if __name__ == "__main__":
    if inductor_indirect_memory_mode:
        run_tests()
