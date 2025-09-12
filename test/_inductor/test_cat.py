import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestCat(TestUtils):

    def op_calc(self, input_element, dim):
        return torch.cat([input_element, input_element], dim)

    # case：change shapes
    @parametrize('shape', [(8, 16, 32, 64)])
    @parametrize('dim', [-1])
    @parametrize('dtype', ['bfloat16'])
    def test_reduction_cases_shapes(self, shape, dim, dtype):
        input_element = self._generate_tensor(shape, dtype)
        std_cat = self.op_calc(input_element, dim)
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_cat = compiled_op_calc(input_element, dim)
        self.assertEqual(std_cat, inductor_cat, atol=1e-1, rtol=1e-1, equal_nan=True)

    def op_calc_non_contiguous(self, input_element, dim):
        return torch.cat([input_element, input_element], dim)

    @parametrize('shape', [(8, 16, 32, 64)])
    @parametrize('dim', [1])
    @parametrize('dtype', ['bfloat16'])
    def test_cat_non_contiguous(self, shape, dim, dtype):
        input_element = self._generate_tensor(shape, dtype)
        input_element = input_element.transpose(-1, -2)
        std_cat = self.op_calc_non_contiguous(input_element, dim)
        compiled_op_calc = torch.compile(self.op_calc_non_contiguous, backend="inductor")
        inductor_cat = compiled_op_calc(input_element, dim)
        self.assertEqual(std_cat, inductor_cat, atol=1e-4, rtol=1e-4, equal_nan=True)

    class PatternModel(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, *xs):
            slices = [x[..., :sz] for x, sz in zip(xs, (128, 32, 48, 48, 48, 48, 48))]
            output_tensor = torch.cat(slices, self.dim)

            return output_tensor
    
    @parametrize('shape', [(128, 50, 128)])
    @parametrize('dim', [2])
    @parametrize('dtype', ['float32', 'bfloat16'])
    def test_model_input_is_concat(self, shape, dim, dtype):
        inputs = [self._generate_tensor(shape, dtype) for _ in range(7)]

        model = self.PatternModel(dim).to(dtype=getattr(torch, dtype))
        model.eval()
        with torch.no_grad():
            eager_out = model(*inputs)

        compiled_model = torch.compile(model, backend="inductor")
        with torch.no_grad():
            inductor_out = compiled_model(*inputs)

        self.assertEqual(eager_out, inductor_out,
                        atol=1e-4, rtol=1e-4, equal_nan=True)

instantiate_parametrized_tests(TestCat)

if __name__ == "__main__":
    run_tests()
