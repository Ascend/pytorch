from torch.library import triton_op, wrap_triton
import torch
import triton
from triton import language as tl
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils
import torch_npu


class TestWrapTriton(TestUtils):

    @staticmethod
    @triton.jit
    def sin_kernel(in_ptr0, out_ptr, n_elements, BLOCK_SIZE: "tl.constexpr"):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        output = tl.sin(x)
        tl.store(out_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton_op("mylib::mysin", mutates_args={})
    def mysin(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n_elements = x.numel()
        wrap_triton(TestWrapTriton.sin_kernel)[(n_elements,)](
            x, out, n_elements, BLOCK_SIZE=4
        )
        return out

    def op_calc(self, x):
        return self.mysin(x) + x

    @parametrize('shape', [(3,)])
    def test_wrap_triton(self, shape):
        x = torch.randn(shape, requires_grad=False, dtype=torch.float32, device="npu")
        std_out = self.op_calc(x)
        compile_out, codes = run_and_get_code(torch.compile(self.op_calc), x)
        self.assertEqual(std_out, compile_out, atol=1e-5, rtol=1e-5)
        self.assertTrue('sin_kernel_0.run' in codes[0])


instantiate_parametrized_tests(TestWrapTriton)

if __name__ == "__main__":
    run_tests()