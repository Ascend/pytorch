import torch
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu._inductor


class TestContiguous(TestCase):
    def test_contiguous_lowering_noncontiguous_input(self):
        torch_npu._inductor._load_triton_backend()

        def make_contiguous(x):
            return (torch.ops.aten.contiguous.default(x),)

        inputs = (
            torch.arange(180, dtype=torch.float16, device="npu")
            .reshape(3, 5, 3, 4)
            .permute(0, 2, 1, 3),
        )
        self.assertFalse(inputs[0].is_contiguous())
        # Keep contiguous explicit so this exercises lowering rather than tracing.
        module = torch.fx.symbolic_trace(make_contiguous)

        def compile_and_run():
            compiled = compile_fx_inner(module, inputs, is_inference=True)
            return compiled(list(inputs))

        with torch.no_grad():
            expected = inputs[0].contiguous()
            (actual,), codes = run_and_get_code(compile_and_run)

        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(actual.stride(), (60, 20, 4, 1))
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(codes)
        self.assertNotIn("torch.ops.aten.contiguous.default(", "\n".join(codes))


if __name__ == "__main__":
    run_tests()
