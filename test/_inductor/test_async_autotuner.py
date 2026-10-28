import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestAsyncAutotuner(TestUtils):

    def op_calc_mm(self, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12):
        addmm_1 = torch.ops.aten.addmm.default(arg_1, arg_2, arg_3)
        addmm_2 = torch.ops.aten.addmm.default(addmm_1, arg_4, arg_5)
        addmm_3 = torch.ops.aten.addmm.default(addmm_2, arg_6, arg_7)
        addmm_4 = torch.ops.aten.addmm.default(addmm_3, arg_8, arg_9)
        permute_1 = torch.ops.aten.permute.default(arg_10, [1, 0])
        relu_1 = torch.ops.aten.relu.default(addmm_4)
        add_2 = torch.ops.aten.add.Tensor(permute_1, relu_1)
        addmm_5 = torch.ops.aten.addmm.default(add_2, arg_11, arg_12)
        return addmm_5


    @parametrize('dtype', ['float32'])
    def test_asyncautotuner(self, dtype):
        M, N, K = 128, 64, 32
        arg_1 = torch.randn(N, device="npu")
        arg_2 = torch.randn(M, K, device="npu")
        arg_3 = torch.randn(K, N, device="npu")
        arg_4 = torch.randn(M, K, device="npu")
        arg_5 = torch.randn(K, N, device="npu")
        arg_6 = torch.randn(M, K, device="npu")
        arg_7 = torch.randn(K, N, device="npu")
        arg_8 = torch.randn(M, K, device="npu")
        arg_9 = torch.randn(K, N, device="npu")
        arg_10 = torch.randn(N, M, device="npu")
        arg_11 = torch.randn(M, K, device="npu")
        arg_12 = torch.randn(K, N, device="npu")
        args = (
            arg_1, arg_2, arg_3, arg_4,
            arg_5, arg_6, arg_7, arg_8,
            arg_9, arg_10, arg_11, arg_12
        )

        std_result = self.op_calc_mm(*args)
        with torch.no_grad():
            compiled_op_calc = torch.compile(self.op_calc_mm, backend="inductor")
            inductor_result = compiled_op_calc(*args)
            self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)

instantiate_parametrized_tests(TestAsyncAutotuner)

if __name__ == "__main__":
    run_tests()
