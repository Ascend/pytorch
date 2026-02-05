import torch
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils
import torch_npu


class TestCurrentDevice(TestUtils):

    @parametrize('shape', [(2, 2)])
    @parametrize('dtype', ['float32'])
    def test_current_device(self, shape, dtype):
        def op_calc(x, y):
            x = x * y
            t = torch.empty(
                size=x.shape,
                dtype=x.dtype,
                device=torch.npu.current_device(),
            )
            out = x + y
            return out

        x = self._generate_tensor(shape, dtype)
        y = self._generate_tensor(shape, dtype)
        compile_result, codes = run_and_get_code(torch.compile(op_calc, backend='inductor'), x, y)
        self.assertTrue('triton_unk_fused_add_mul_0.run' in codes[0])


instantiate_parametrized_tests(TestCurrentDevice)


if __name__ == "__main__":
    run_tests()
