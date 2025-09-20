import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestEq(TestUtils):
    def op_calc(self, first_element, second_element):
        return torch.eq(first_element, second_element)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float32', 'int32', 'float16', 'bfloat16'])
    def test_pointwise_cases(self, shape, dtype):

        first_element = self._generate_tensor(shape, dtype)
        second_element = first_element.clone()

        # randomly change some elements in second tensor
        flat_second_view = second_element.flatten()
        num_elements_to_change = first_element.numel() // 3
        random_indices = torch.randint(0, first_element.numel(), (num_elements_to_change,))
        flat_second_view[random_indices] = 1 - flat_second_view[random_indices]

        std_result = self.op_calc(first_element, second_element)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, second_element)

        self.assertEqual(std_result, inductor_result)


instantiate_parametrized_tests(TestEq)

if __name__ == "__main__":
    run_tests()
