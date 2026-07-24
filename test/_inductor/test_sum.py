import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestSum(TestUtils):
    def op_calc(self, input_element, dim):
        return torch.sum(input_element, dim)

    # 规约轴和非规约轴对齐用例 float32 XBLOCK_SUB>=8:shape=(8,32)
    # non-persistent reduction 用例 规约轴>1024:shape=(8,8,8,2048) dim=-1
    _reduction_extest_shape4d_all = [(8, 32), (8, 8, 8, 2048)]
    _reduction_extest_dim4d_low = [-1]
    _reduction_extest_dim4d_all = [0, 1, 2]

    @parametrize('shape', _reduction_extest_shape4d_all)
    @parametrize('dim', _reduction_extest_dim4d_low)
    @parametrize('dtype', ['float32'])
    def test_reduction_cases_shapes(self, shape, dim, dtype):
        input_element = self._generate_tensor(shape, dtype)
        std_sum = self.op_calc(input_element, dim)
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor", dynamic=False)
        inductor_sum_tmp = compiled_op_calc(input_element, dim)
        if dtype == 'int32' or dtype == 'int64':
            # inductor return float32,need to change int64 for assert
            inductor_sum = inductor_sum_tmp.long()
        elif dtype == 'float16':
            # inductor return float32,need to change float16 for assert
            inductor_sum = inductor_sum_tmp.half()
        elif dtype == 'bfloat16':
            # inductor return float32,need to change float32 for assert
            std_sum = std_sum.float()
            inductor_sum = inductor_sum_tmp
        else:
            inductor_sum = inductor_sum_tmp

        self.assertEqual(std_sum, inductor_sum, atol=1e-1, rtol=1e-1)

    @parametrize('shape', [(8193, 1)])
    @parametrize('dim', [None])
    @parametrize('dtype', ['int32'])
    def test_sum_all_reduction_runtime_rblock(self, shape, dim, dtype):
        input_element = torch.ones(shape, dtype=eval('torch.' + dtype), device=torch.device("npu"))
        std_sum = self.op_calc(input_element, dim)
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor", dynamic=False)
        compiled_op_calc(input_element, dim)
        inductor_sum = compiled_op_calc(input_element, dim)

        self.assertEqual(std_sum, inductor_sum)

    @parametrize('shape', [(32, 16, 64, 128)])
    @parametrize('dim', _reduction_extest_dim4d_all)
    @parametrize('dtype', ['float32'])
    def test_reduction_cases_dims(self, shape, dim, dtype):

        input_element = self._generate_tensor(shape, dtype)
        std_sum = self.op_calc(input_element, dim)
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor", dynamic=False)
        inductor_sum = compiled_op_calc(input_element, dim)

        self.assertEqual(std_sum, inductor_sum, atol=1e-1, rtol=1e-1)

    # Sizes straddle bucket boundaries plus non-power-of-2 / large, to exercise
    # the runtime loop bound + tail mask.
    _sum_1d_dynamic_sizes = [8, 63, 64, 65, 255, 257, 1024, 1025, 8191, 8193, 50000, 999983]

    @parametrize('dtype', ['float32'])
    def test_sum_1d_dynamic_shape(self, dtype):
        # One dynamic kernel must stay correct across all sizes without recompiling.
        torch_dtype = eval('torch.' + dtype)
        torch._dynamo.reset()
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor", dynamic=True)
        for n in self._sum_1d_dynamic_sizes:
            x = torch.ones((n,), dtype=torch_dtype, device=torch.device("npu"))
            std_sum = self.op_calc(x, None)
            inductor_sum = compiled_op_calc(x, None)
            self.assertEqual(std_sum, inductor_sum)

    @parametrize('dtype', ['float32'])
    def test_sum_1d_dynamic_shape_group_autotune(self, dtype):
        # Same, with symbolic group-autotune (bucketed path) enabled.
        import torch_npu._inductor.config as npu_config
        torch_dtype = eval('torch.' + dtype)
        prev = npu_config.enable_symbolic_shape_group_autotune
        npu_config.enable_symbolic_shape_group_autotune = True
        try:
            torch._dynamo.reset()
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor", dynamic=True)
            for n in self._sum_1d_dynamic_sizes:
                x = torch.ones((n,), dtype=torch_dtype, device=torch.device("npu"))
                std_sum = self.op_calc(x, None)
                inductor_sum = compiled_op_calc(x, None)
                self.assertEqual(std_sum, inductor_sum)
        finally:
            npu_config.enable_symbolic_shape_group_autotune = prev
            torch._dynamo.reset()


instantiate_parametrized_tests(TestSum)

if __name__ == "__main__":
    run_tests()
