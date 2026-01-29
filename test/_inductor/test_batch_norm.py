import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestNativeBatchNorm(TestUtils):
    def op_calc(self, input_element):
        # 创建权重和偏置张量
        weight = torch.ones(32).npu()
        bias = torch.zeros(32).npu()

        # 创建运行均值和方差张量
        running_mean = torch.zeros(32).npu()
        running_var = torch.ones(32).npu()
        momentum = 0.1
        eps = 1e-05
        # 执行批量归一化
        output, running_mean_out, running_var_out = torch.native_batch_norm(
            input=input_element,
            weight=weight,
            bias=bias,
            running_mean=running_mean,
            running_var=running_var,
            training=True,
            momentum=momentum,
            eps=eps
        )
        return output, running_mean_out, running_var_out

    @parametrize('shape', [(16, 32, 64)])
    @parametrize('dtype', ['float32'])
    def test_reduction_cases_shapes(self, shape, dtype):
        input_element = self._generate_tensor(shape, dtype)

        std_ret, _, _ = self.op_calc(input_element)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_ret, _, _ = compiled_op_calc(input_element)
        self.assertEqual(std_ret, inductor_ret, atol=1e-1, rtol=1e-1, equal_nan=True)


instantiate_parametrized_tests(TestNativeBatchNorm)

if __name__ == "__main__":
    run_tests()
