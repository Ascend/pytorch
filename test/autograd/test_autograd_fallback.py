import torch
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)
import torch_npu

class TestAutogradFallback(TestCase):

    def test_pad_backward_warn(self):

        def _exec_npu_pad():
            npu_input = torch.randn(2, 3).npu()
            npu_input.requires_grad = True
            pads = (1, 1, 1, 1)
            output = torch_npu.npu_pad(npu_input, pads)
            output.backward(torch.ones_like(output))

        # When set to "nothing," calling the reverse function directly causes an error.
        torch._C._set_autograd_fallback_mode("nothing")
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            _exec_npu_pad()

        # When set to "warn," calling the print function emits a warning.
        torch._C._set_autograd_fallback_mode("warn")
        _exec_npu_pad()


if __name__ == "__main__":
    run_tests()
