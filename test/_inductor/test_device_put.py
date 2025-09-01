import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestDevicePut(TestUtils):
    def op_calc(self, input_element1, input_element2):
        return torch.add(input_element1, input_element2)

    @parametrize('shape', [(8, 16, 8)])
    @parametrize('dtype', ['int32'])
    def test_cases_shapes(self, shape, dtype):
        low = 0
        high = 2
        dtype = eval('torch.' + dtype)
        npu_device = torch.device('npu:0')
        input_element1_tmp = torch.randint(low, high, shape, dtype=dtype).cpu()
        input_element2_tmp = torch.randint(low, high, shape, dtype=dtype).cpu()
        input_element1 = torch.ops.prims.device_put(input_element1_tmp, npu_device)
        input_element2 = torch.ops.prims.device_put(input_element2_tmp, npu_device)

        std_ret = self.op_calc(input_element1, input_element2)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_ret = compiled_op_calc(input_element1, input_element2)

        self.assertEqual(std_ret, inductor_ret)


instantiate_parametrized_tests(TestDevicePut)

if __name__ == "__main__":
    run_tests()
