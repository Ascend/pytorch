import torch
import triton
import triton.language as tl
from triton.language.extra.cann import libdevice
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch._inductor.utils import run_and_get_code


@triton.jit
def softcap_fwd_kernel(x_ptr, y_ptr, n_elements, softcap, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)
    y = softcap * libdevice.tanh(x_f32 / softcap)
    y = y.to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)


def softcap_forward(x, softcap):
    numel = x.numel()
    y = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"]),)
    softcap_fwd_kernel[grid](x, y, numel, softcap, BLOCK_SIZE=2048)
    return y + x


class TestDvmWrap(TestCase):
    def test_softcap_dvm(self):
        dtype = torch.float16
        softcap = 50.0
        x = torch.randn((1024, 1024), dtype=dtype, device="npu")
        std_out = softcap_forward(x, softcap)
        fn = torch.compile(softcap_forward, options={"npu_backend": "dvm"})
        compile_out, codes = run_and_get_code(fn, x, softcap)
        self.assertEqual(std_out, compile_out, atol=1e-3, rtol=1e-3)
        self.assertIn("softcap_fwd_kernel", codes[0])
        self.assertIn("dvm", codes[0])

instantiate_parametrized_tests(TestDvmWrap)
if __name__ == "__main__":
    run_tests()
