import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
from torch_npu._inductor.config import arch_support_simt
from torch_npu._inductor.npu_triton_heuristics import do_bench_using_profiling_npu


IndexSelectParamInfo = [
    [(10001, 128), (128,)],
    [(10001, 64), (128, )],
    [(10001, 32), (128, )],
    [(4, 128), (128, )],
    [(2048, 16), (20480, )],
    [(2048, 16), (1063, )]
]


class TestAtenIndexSelectSimt(TestUtils):
    def index_select(self, table, index):
        return torch.index_select(table, 1, index)

    @parametrize('shape_info', IndexSelectParamInfo)
    def test_aten_index_select(self, shape_info, enable_profiling=False):
        table_shape, index_shape = shape_info
        index = torch.randint(0, table_shape[0], size=index_shape, dtype=torch.int64).npu()
        table = torch.randn(table_shape, requires_grad=False, dtype=torch.float32, device='npu')

        index_select_triton = torch.compile(self.index_select, backend="inductor", dynamic=False)

        def eager_fn():
            return self.index_select(table, index)
        
        def inductor_fn():
            return index_select_triton(table, index)

        r = eager_fn()
        r1 = inductor_fn()
        self.assertEqual(r, r1)

        if enable_profiling:
            eager_time = do_bench_using_profiling_npu(eager_fn)
            inductor_time = do_bench_using_profiling_npu(inductor_fn)
            print(f"Gather benchmark, table_shape: {table_shape}, index shape: {index_shape}, eager: {eager_time}, inductor_time: {inductor_time}")

instantiate_parametrized_tests(TestAtenIndexSelectSimt)

if __name__ == "__main__":
    if arch_support_simt:
        run_tests()