import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils


class TestAnyProdCodegen(TestUtils):

    @parametrize('shape', [(16, 32, 64)])
    @parametrize('dtype', ['int32'])
    def test_aten_any(self, shape, dtype):
        def op_calc(first_element):
            x = first_element + 1
            return torch.ops.aten.any(x)

        input_element = self._generate_tensor(shape, dtype)
        std_ret = op_calc(input_element)
        compiled_op_calc = torch.compile(op_calc, backend="inductor", dynamic=False)
        inductor_ret = compiled_op_calc(input_element)
        self.assertEqual(std_ret, inductor_ret)


    @parametrize('shape', [(16, 32, 64)])
    @parametrize('dim', [0, 1, -1])
    @parametrize('dtype', ['int32'])
    def test_aten_any_dim(self, shape, dtype, dim):
        def op_calc(first_element):
            x = first_element + 1
            return torch.ops.aten.any(x, dim=dim)

        input_element = self._generate_tensor(shape, dtype)
        std_ret = op_calc(input_element)
        compiled_op_calc = torch.compile(op_calc, backend="inductor", dynamic=False)
        inductor_ret = compiled_op_calc(input_element)
        self.assertEqual(std_ret, inductor_ret)



instantiate_parametrized_tests(TestAnyProdCodegen)

if __name__ == "__main__":
    run_tests()
