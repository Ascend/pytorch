import torch
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu._inductor


class TestContiguous(TestCase):
    def test_contiguous_lowering_after_index_select(self):
        torch_npu._inductor._load_triton_backend()

        def gather(x, index):
            x = torch.ops.aten.permute.default(x, [0, 2, 1, 3])
            x = torch.ops.aten.index.Tensor(x, [index])
            return (torch.ops.aten.contiguous.default(x),)

        inputs = (
            torch.arange(120, dtype=torch.float16, device="npu").reshape(2, 5, 3, 4),
            torch.tensor([1, 0, 1], device="npu"),
        )
        # Keep contiguous explicit so this exercises lowering rather than tracing.
        module = torch.fx.symbolic_trace(gather)

        def compile_and_run():
            compiled = compile_fx_inner(module, inputs, is_inference=True)
            return compiled(list(inputs))

        with torch.no_grad():
            expected = inputs[0].permute(0, 2, 1, 3).index_select(0, inputs[1])
            (actual,), codes = run_and_get_code(compile_and_run)

        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(actual.stride(), (60, 20, 4, 1))
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(codes)
        self.assertNotIn("torch.ops.aten.contiguous.default(", "\n".join(codes))


if __name__ == "__main__":
    run_tests()
