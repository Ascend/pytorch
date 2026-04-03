import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu

class TestFrexp(TestUtils):
    @parametrize('shape', [(4,3)])
    @parametrize('dtype', ['float32'])
    def test_aten_frexp(self, shape, dtype):
        def op_calc(first_element):
            mantissa, exponent = torch.ops.aten.frexp(first_element)
            mantissa =mantissa +1
            exponent =exponent +1
            return mantissa, exponent        
        print(f"=========================================aten.frexp====={dtype}=================================================")
        first_element = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device='npu', dtype = eval('torch.' + dtype))
        print(f"\nfirst_element\n{first_element}")
        std_mantissa,  std_exponent= op_calc(first_element)

        compiled_op_calc = torch.compile(op_calc, backend="inductor")
        inductor_mantissa, inductor_exponent = compiled_op_calc(first_element)
        print(f"\nstd_mantissa\n{std_mantissa}")
        print(f"\ninductor_mantissa\n{inductor_mantissa}")
        # self.assertEqual(std_mantissa, inductor_mantissa)

        print(f"\nstd_exponent\n{std_exponent}")
        print(f"\ninductor_exponent\n{inductor_exponent}")
        self.assertEqual(std_exponent, inductor_exponent)

    @parametrize('shape', [(4,3)])
    @parametrize('dtype', ['bfloat16','float16','float32'])
    def test_prims_frexp(self, shape, dtype):
        def op_calc(first_element):
            mantissa, exponent = torch.ops.prims.frexp.default(first_element)
            mantissa =mantissa +1
            exponent =exponent +1
            return mantissa, exponent        
        print(f"=========================================prims.frexp====={dtype}=================================================")
        first_element = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], device='npu', dtype = eval('torch.' + dtype))
        print(f"\nfirst_element\n{first_element}")
        std_mantissa,  std_exponent= op_calc(first_element)

        compiled_op_calc = torch.compile(op_calc, backend="inductor")
        inductor_mantissa, inductor_exponent = compiled_op_calc(first_element)
        print(f"\nstd_mantissa\n{std_mantissa}")
        print(f"\ninductor_mantissa\n{inductor_mantissa}")
        # self.assertEqual(std_mantissa, inductor_mantissa)

        print(f"\nstd_exponent\n{std_exponent}")
        print(f"\ninductor_exponent\n{inductor_exponent}")
        self.assertEqual(std_exponent, inductor_exponent)


instantiate_parametrized_tests(TestFrexp)

if __name__ == "__main__":
    run_tests()