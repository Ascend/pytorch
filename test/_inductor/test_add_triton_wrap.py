from torch.library import triton_op, wrap_triton
import torch
import triton
from triton import language as t1
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
    def sin_kernel(in_ptr0, out_ptr, n_elements, BLOCK_SIZE: "t1.constexpr"):
        pid = t1.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + t1.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = t1.load(in_ptr0 + offsets, mask=mask)
        output = t1.sin(x)
        t1.store(out_ptr + offsets, output, mask=mask)

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
        compile_out = torch.compile(self.op_calc)(x)
        self.assertEqual(std_out, compile_out, atol=1e-5, rtol=1e-5)


instantiate_parametrized_tests(TestWrapTriton)

if __name__ == "__main__":
    run_tests()
