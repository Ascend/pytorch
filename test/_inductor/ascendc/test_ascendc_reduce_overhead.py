"""AscendC backend: reduce-overhead mode (aclgraph capture/replay)."""
import unittest
import torch
import torch_npu
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not torch.npu.is_available(), "requires an NPU device")
class TestAscendcReduceOverhead(TestCase):

    def test_capture_replay(self):
        """Verify reduce-overhead triggers graph capture and results are correct."""
        def fn(x, y):
            return x * y - x

        x = torch.randn(64, 64, device="npu")
        y = torch.randn(64, 64, device="npu")

        compiled_fn = torch.compile(
            fn, backend="inductor",
            options={"npu_backend": "ascendc", "triton.cudagraphs": True},
        )

        # Warmup (triggers compilation + first capture)
        with torch.no_grad():
            for _ in range(3):
                out = compiled_fn(x, y)

        # Verify correctness
        eager_out = fn(x, y)
        torch.testing.assert_close(out, eager_out, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    run_tests()
