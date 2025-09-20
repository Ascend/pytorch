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

    @parametrize('shape', [(32, 16, 64, 128)])
    @parametrize('dim', _reduction_extest_dim4d_all)
    @parametrize('dtype', ['float32'])
    def test_reduction_cases_dims(self, shape, dim, dtype):

        input_element = self._generate_tensor(shape, dtype)
        std_sum = self.op_calc(input_element, dim)
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor", dynamic=False)
        inductor_sum = compiled_op_calc(input_element, dim)

        self.assertEqual(std_sum, inductor_sum, atol=1e-1, rtol=1e-1)


instantiate_parametrized_tests(TestSum)

if __name__ == "__main__":
    run_tests()
