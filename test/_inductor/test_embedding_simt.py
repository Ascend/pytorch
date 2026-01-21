import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
from torch_npu._inductor.config import inductor_indirect_memory_mode
from torch_npu._inductor.npu_triton_heuristics import do_bench_using_profiling_npu

# table_shape, index_shape
EmbeddingParamInfo = [
    [(10001, 128), (128, 50)],
    [(10001, 64), (128, 50)],
    [(10001, 32), (128, 50)],
    [(10001, 16), (128, 50)],
    [(10001, 8), (128, )],
    [(315511, 32), (128, )],
]

# table_shape, index_shape
EmbeddingParamInfoBenchmark1 = [
    [(10001, 128), (128, 50)],
    [(10001, 64), (128, 50)],
    [(10001, 32), (128, 50)],
    [(10001, 16), (128, 50)],
    [(10001, 8), (128, 50)],
    [(1353406, 8), (128, 50)],
    [(315511, 32), (128, 50)],
    [(81913, 64), (128, 38)],
    [(179390, 16), (128, )],
    [(1820039, 8), (128, )],
    [(289094, 64), (128, )],
    [(293512, 32), (128, )],
    [(115538, 32), (128, )],
    [(8202, 128), (128, )],
    [(7335, 8), (128, )],
    [(4, 128), (128, )],
    [(5, 8), (128, )],
    [(51, 64), (128, )],
    [(8, 16), (128, )],
    [(312185, 16), (312185, )],
    [(169219, 16), (169219, )],
    [(2048, 16), (2048, )],
    [(2048, 16), (1989, )],
    [(5, 8), (2, 128)],
    [(10001, 128), (2, 100)],
]

EmbeddingParamInfoBenchmark2 = [
    [(324178, 16), (324178, )],
    [(324178, 16), (180108, )],
    [(186255, 16), (182379, )],
    [(9000, 128), (9000, )],
    [(9000, 128), (4000, )],
    [(9000, 128), (1024, )],
    [(2048, 16), (20480, )],
    [(2048, 16), (1063, )],
    [(4, 128), (128, )],
    [(324178, 16), (324178,)],
    [(324178, 16), (180108,)],
    [(186255, 16), (182379, )],
    [(9000, 128), (9000, )],
    [(2048, 16), (20480, )],
    [(12800, 128), (2056, )],
    [(64, 64), (6,)],
    [(256, 256), (25,)],
    [(1024, 1024), (102,)],
    [(1024, 65536), (102,)],
    [(4096, 4096), (409,)],
]


class TestAtenEmbeddingSimt(TestUtils):
    def embedding(self, embedding_table, index):
        return torch.ops.aten.embedding.default(embedding_table, index)

    @parametrize('shape_info', EmbeddingParamInfo)
    def test_aten_embedding(self, shape_info, enable_profiling=False):
        [table_shape, index_shape] = shape_info
        embedding_vocabulary = table_shape[0]
        index = torch.randint(0, embedding_vocabulary, size=index_shape, dtype=torch.int32).npu()
        embedding_table = torch.randn(table_shape, requires_grad=False, dtype=torch.float32, device='npu')
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
            print(f"Embedding benchmark, table_shape: {table_shape}, index shape: {index_shape}, eager: {eager_time}, inductor_time: {inductor_time}")

instantiate_parametrized_tests(TestAtenEmbeddingSimt)

if __name__ == "__main__":
    if inductor_indirect_memory_mode:
        run_tests()