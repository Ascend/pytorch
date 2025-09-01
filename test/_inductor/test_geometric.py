import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestGeometric(TestUtils):
    def op_calc(self):
        # 创建一个形状为 (3, 3)的张量， 每个位置的概率为 0.5
        prob = torch.full((16, 16), 0.5).npu()

        #使用 aten.geometric生成几何分布的随机数
        geometric_tensor = torch.ops.aten.geometric(prob, p=0.5)

        return geometric_tensor

    # UT skip, reason: this has problem in torch 260
    # Added to pytorch-disable-tests.json
    @parametrize('shape', [(16, 16, 16)])
    @parametrize('dim', [0])
    @parametrize('dtype', ['int32'])
    def test_reduction_cases_shapes(self, shape, dim, dtype):
        std_ret = self.op_calc()
        std_ret_mean = torch.mean(std_ret)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_ret = compiled_op_calc()

        inductor_ret_mean = torch.mean(inductor_ret)
        self.assertTrue(inductor_ret_mean is not None)


instantiate_parametrized_tests(TestGeometric)

if __name__ == "__main__":
    run_tests()
