import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestSumAdd(TestUtils):
    def op_calc(self, input_element, dim, input_element2):
        tmp = torch.sum(input_element, dim)
        return tmp + input_element2

    @parametrize('shape', [(32, 64, 128, 2048)])
    @parametrize('dim', [0, 1, 2, 3])
    @parametrize('dtype', ['float32'])
    def test_reduction_cases_shapes(self, shape, dim, dtype):
        input_element = self._generate_tensor(shape, dtype)
        if dim == -1 or dim == 3:
            input_element2 = torch.full(size=(32, 64, 128), fill_value=1000.0, dtype=torch.float32, device=torch.device("npu"))
        elif dim == 2:
            input_element2 = torch.full(size=(32, 64, 2048), fill_value=1000.0, dtype=torch.float32, device=torch.device("npu"))
        elif dim == 1:
            input_element2 = torch.full(size=(32, 128, 2048), fill_value=1000.0, dtype=torch.float32, device=torch.device("npu"))
        else:
            input_element2 = torch.full(size=(64, 128, 2048), fill_value=1000.0, dtype=torch.float32, device=torch.device("npu"))

        std_sum = self.op_calc(input_element, dim, input_element2)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_sum = compiled_op_calc(input_element, dim, input_element2)

        self.assertEqual(std_sum, inductor_sum, atol=1e-1, rtol=1e-1)


instantiate_parametrized_tests(TestSumAdd)

if __name__ == "__main__":
    run_tests()
