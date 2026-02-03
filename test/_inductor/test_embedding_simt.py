import unittest
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

    def do_single_embedding_test(self, param_info, index_dtype, table_dtype):
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

    @parametrize('param_info', EmbeddingParamInfo)
    @parametrize('index_dtype', IndexDtype)
    @parametrize('table_dtype', TableDtype)
    def test_aten_embedding(self, param_info, index_dtype, table_dtype):
        self.do_single_embedding_test(param_info, index_dtype, table_dtype)

    @parametrize('param_info', EmbeddingParamInfoBenchmark)
    @parametrize('index_dtype', IndexDtype)
    @parametrize('table_dtype', TableDtype)
    def test_aten_embedding_benchmark(self, param_info, index_dtype, table_dtype):
        self.do_single_embedding_test(param_info, index_dtype, table_dtype)

    def test_aten_embedding_bound_check(self):
        [table_shape, index_shape, enable_profiling] = [(10001, 128), (128, 50), False]
        index_dtype, table_dtype = torch.int64, torch.float32
        index = -torch.randint(0, table_shape[0], size=index_shape, dtype=index_dtype, device='npu')
        embedding_table = torch.randn(table_shape, requires_grad=False, dtype=table_dtype, device='npu')
        embedding_triton = torch.compile(self.embedding, backend="inductor", dynamic=False)

        def eager_fn():
            index_pos = index + table_shape[0]
            return self.embedding(embedding_table, index_pos)

        def inductor_fn():
            return embedding_triton(embedding_table, index)

        r = eager_fn()
        r1 = inductor_fn()
        self.assertEqual(r, r1)

    def test_embedding_unsqueeze_expand_dim1(self):

        def embedding_view(table, dim, indice):
            indice = indice.expand(1, 128)
            indice = indice.reshape(128, )
            table = table.unsqueeze(-1)
            table = table.expand(16, 1024)
            table = table.permute((1, 0))
            return torch.ops.aten.embedding.default(table, indice)

        table_shape = (1024, 16)
        index_shape = (128, )
        table_dtype, index_dtype = torch.float32, torch.int64
        dim = 0
        indice = torch.randint(0, table_shape[dim], size=(1, ), dtype=index_dtype, device='npu')
        table = torch.randn((16, ), requires_grad=False, dtype=table_dtype, device='npu')
        embedding_triton = torch.compile(embedding_view, backend="inductor", dynamic=False)

        r = embedding_view(table, dim, indice)
        r1 = embedding_triton(table, dim, indice)
        self.assertEqual(r, r1)

    def test_embedding_unsqueeze_expand_dim2(self):

        def embedding_view(table, dim, indice):
            indice = indice.expand(128, 16)
            table = table.unsqueeze(-1)
            table = table.expand(16, 1024)
            table = table.permute((1, 0))
            return torch.ops.aten.embedding.default(table, indice)

        table_shape = (1024, 16)
        index_shape = (128, 16)
        table_dtype, index_dtype = torch.float32, torch.int64
        dim = 0
        indice = torch.randint(0, table_shape[dim], size=(128, 1), dtype=index_dtype, device='npu')
        table = torch.randn((16, ), requires_grad=False, dtype=table_dtype, device='npu')
        embedding_triton = torch.compile(embedding_view, backend="inductor", dynamic=False)

        r = embedding_view(table, dim, indice)
        r1 = embedding_triton(table, dim, indice)
        self.assertEqual(r, r1)

    def test_embedding_unsqueeze_expand_pointwise(self):

        def embedding_view(table, dim, indice, table2):
            indice = indice.expand(1, 128)
            indice = indice.reshape(128, )
            table = table.unsqueeze(-1)
            table = table.expand(16, 1024)
            table = table.permute((1, 0))
            table = table + table2
            return torch.ops.aten.embedding.default(table, indice)

        table_shape = (1024, 16)
        index_shape = (128, )
        table_dtype, index_dtype = torch.float32, torch.int64
        dim = 0
        indice = torch.randint(0, table_shape[dim], size=(1, ), dtype=index_dtype, device='npu')
        table = torch.randn((16, ), requires_grad=False, dtype=table_dtype, device='npu')
        table2 = torch.randn((16, ), requires_grad=False, dtype=table_dtype, device='npu')
        embedding_triton = torch.compile(embedding_view, backend="inductor", dynamic=False)

        r = embedding_view(table, dim, indice, table2)
        r1 = embedding_triton(table, dim, indice, table2)
        self.assertEqual(r, r1)


    def test_embedding_slice_cat(self):

        def embedding_slice_cat(table, indices):
            indice1 = torch.ops.aten.slice.Tensor(indices, 1, 0, 20)
            indice2 = torch.ops.aten.slice.Tensor(indices, 1, 20, 40)
            indice3 = torch.ops.aten.slice.Tensor(indices, 1, 40, 60)
            embedding1 = torch.ops.aten.embedding.default(table, indice1)
            embedding2 = torch.ops.aten.embedding.default(table, indice2)
            embedding3 = torch.ops.aten.embedding.default(table, indice3)
            sum1 = torch.ops.aten.sum.dim_IntList(embedding1, [1])
            sum2 = torch.ops.aten.sum.dim_IntList(embedding2, [1])
            sum3 = torch.ops.aten.sum.dim_IntList(embedding3, [1])
            return torch.ops.aten.cat.default([embedding1, embedding2, embedding3], 1)

        table_shape = (5000, 8)
        index_shape = (128, 60)
        table_dtype, index_dtype = torch.float32, torch.int64
        dim = 0
        indice = torch.randint(0, table_shape[dim], size=index_shape, dtype=index_dtype, device='npu')
        table = torch.randn(table_shape, requires_grad=False, dtype=table_dtype, device='npu')
        embedding_triton = torch.compile(embedding_slice_cat, backend="inductor", dynamic=False)

        r = embedding_slice_cat(table, indice)
        r1 = embedding_triton(table, indice)
        self.assertEqual(r, r1)



instantiate_parametrized_tests(TestAtenEmbeddingSimt)

if __name__ == "__main__":
    if inductor_indirect_memory_mode:
        run_tests()
