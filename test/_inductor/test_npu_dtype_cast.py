import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestNpuDtypeCast(TestUtils):
    def op_calc(self, first_element, second_element):
        x = first_element.to(torch.float32)
        y = torch.ops.npu._npu_dtype_cast.default(second_element, torch.float32)
        result = x + y
        ret = torch.ops.npu.npu_dtype_cast.default(result, torch.int32)
        return ret

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float16'])
    def test_fuse_npu_dtype_cast(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_tensor(shape, dtype)

        std_sum = self.op_calc(first_element, second_element)
        compiled_func = torch.compile(self.op_calc, backend="inductor")
        inductor_sum, code = torch._inductor.utils.run_and_get_code(compiled_func, first_element, second_element)

        self.assertEqual(std_sum, inductor_sum)

        code = " ".join(code)
        assert_keywords = ["torch.ops.npu._npu_dtype_cast.default(", "torch.ops.npu.npu_dtype_cast.default("]
        for assert_key in assert_keywords:
            self.assertNotIn(assert_key, code)


instantiate_parametrized_tests(TestNpuDtypeCast)

if __name__ == "__main__":
    run_tests()
