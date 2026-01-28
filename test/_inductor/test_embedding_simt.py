import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import BenchmarkTestUtils
from torch_npu._inductor.config import inductor_indirect_memory_mode
from torch_npu._inductor.npu_triton_heuristics import do_bench_using_profiling_npu


# table_shape, index_shape
EmbeddingParamInfo = [
    [(10001, 128), (128, 50), False],
    [(10001, 64), (128, 50), False],
    [(10001, 32), (128, 50), False],
    [(10001, 16), (128, 50), False],
    [(10001, 8), (128, ), False],
    [(315511, 32), (128, ), False],
]

IndexDtype = [torch.int64]
TableDtype = [torch.float32]

# table_shape, index_shape
EmbeddingParamInfoBenchmark = [
    [(10001, 128), (128, 50), True],
    [(10001, 64), (128, 50), True],
    [(10001, 32), (128, 50), True],
    [(10001, 16), (128, 50), True],
    [(10001, 8), (128, 50), True],
    [(1353406, 8), (128, 50), True],
    [(315511, 32), (128, 50), True],
    [(81913, 64), (128, 38), True],
    [(179390, 16), (128, ), True],
    [(1820039, 8), (128, ), True],
    [(289094, 64), (128, ), True],
    [(293512, 32), (128, ), True],
    [(115538, 32), (128, ), True],
    [(8202, 128), (128, ), True],
    [(7335, 8), (128, ), True],
    [(4, 128), (128, ), True],
    [(5, 8), (128, ), True],
    [(51, 64), (128, ), True],
    [(8, 16), (128, ), True],
    [(312185, 16), (312185, ), True],
    [(169219, 16), (169219, ), True],
    [(2048, 16), (2048, ), True],
    [(2048, 16), (1989, ), True],
    [(5, 8), (2, 128), True],
    [(10001, 128), (2, 100), True],
    [(80000, 64), (16, 8, 128), True],
    [(120000, 256), (4, 2, 256), True],
    [(180000, 32), (64, 16, 32), True],
    [(324178, 16), (324178, ), True],
    [(324178, 16), (180108, ), True],
    [(186255, 16), (182379, ), True],
    [(9000, 128), (9000, ), True],
    [(9000, 128), (4000, ), True],
    [(9000, 128), (1024, ), True],
    [(2048, 16), (20480, ), True],
    [(2048, 16), (1063, ), True],
    [(4, 128), (128, ), True],
    [(324178, 16), (324178,), True],
    [(324178, 16), (180108,), True],
    [(186255, 16), (182379, ), True],
    [(9000, 128), (9000, ), True],
    [(2048, 16), (20480, ), True],
    [(12800, 128), (2056, ), True],
    [(64, 64), (6,), True],
    [(256, 256), (25,), True],
    [(1024, 1024), (102,), True],
    [(1024, 65536), (102,), True],
    [(4096, 4096), (409,), True],
]


class TestAtenEmbeddingSimt(BenchmarkTestUtils):
    profiling_file_path = "TestAtenEmbeddingSimt_perf.csv"

    def embedding(self, embedding_table, index):
        return torch.ops.aten.embedding.default(embedding_table, index)

    @parametrize('param_info', EmbeddingParamInfo)
    @parametrize('index_dtype', IndexDtype)
    @parametrize('table_dtype', TableDtype)
    def test_aten_embedding(self, param_info, index_dtype, table_dtype):
        [table_shape, index_shape, enable_profiling] = param_info
        index = torch.randint(0, table_shape[0], size=index_shape, dtype=index_dtype, device='npu')
        embedding_table = torch.randn(table_shape, requires_grad=False, dtype=table_dtype, device='npu')
        embedding_triton = torch.compile(self.embedding, backend="inductor", dynamic=False)

        def eager_fn():
            return self.embedding(embedding_table, index)

        def inductor_fn():
            return embedding_triton(embedding_table, index)

        r = eager_fn()
        r1 = inductor_fn()
        self.assertEqual(r, r1)

        if enable_profiling:
            eager_time = do_bench_using_profiling_npu(eager_fn)
            inductor_time = do_bench_using_profiling_npu(inductor_fn)
            self.write_performance_info(self.profiling_file_path, {
                "op type": "aten.embedding",
                "shape": f"{shape_info}",
                "dtype": f"({index_dtype}, {table_dtype})",
                "eager_time": f"{eager_time:.2f}",
                "inductor_time": f"{inductor_time:.2f}"
            })

instantiate_parametrized_tests(TestAtenEmbeddingSimt)

if __name__ == "__main__":
    if inductor_indirect_memory_mode:
        run_tests()
