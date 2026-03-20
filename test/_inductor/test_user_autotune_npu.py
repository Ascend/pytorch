import torch
import triton
import triton.language as tl
from torch.testing._internal.common_utils import run_tests, TestCase
import torch_npu
import torch_npu._inductor


class TestUserAutotuneNpu(TestCase):
    def test_user_autotune_npu(self):
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 64}),
                triton.Config({"BLOCK_SIZE": 32}),
            ],
            key=["n_elements"],
        )
        @triton.jit
        def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: "tl.constexpr"):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            tl.store(output_ptr + offsets, x + y, mask=mask)

        def add(x, y):
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            add_kernel[grid](x, y, output, n_elements)
            return output

        x = torch.randn(64, device="npu")
        y = torch.randn(64, device="npu")
        expected = x + y

        compiled = torch.compile(add, backend="inductor")
        result = compiled(x, y)

        self.assertTrue(torch.allclose(result, expected, atol=1e-3))


if __name__ == "__main__":
    run_tests()
