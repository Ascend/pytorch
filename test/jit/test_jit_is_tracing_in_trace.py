# Owner(s): ["oncall: jit"]
"""
Add validation cases for torch.jit tracing APIs on NPU:
Strictly validates the functional correctness of torch.jit.is_tracing()
during torch.jit.trace execution.
Note: Script mode already has community test cases in test/test_jit.py,
this case only verifies trace mode.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
import torch_npu

# Get current accelerator device type
device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestJitIsTracing(TestCase):
    def test_is_tracing_returns_true_in_trace_mode(self):
        """
        Validates that torch.jit.is_tracing() returns True during trace recording.

        Note: Direct assertion inside the traced function is not feasible because
        the Python-level tracing flag remains False during Eager-mode execution
        within torch.jit.trace. Instead, this test strictly verifies the API's
        correctness through behavioral validation:

        If is_tracing() correctly evaluates to True during recording, the tracer
        will capture the 'x + 1' branch and permanently bake it into the TorchScript graph.
        We assert the final output to prove this specific path was recorded.
        """
        def my_func(x):
            # The tracer evaluates this condition during the recording phase.
            if torch.jit.is_tracing():
                return x + 1
            else:
                return x - 1

        inp = torch.randn(3, 3).to(device_type)

        # Execute tracing. If is_tracing() works correctly, the 'x + 1' path is recorded.
        traced_func = torch.jit.trace(my_func, inp, check_trace=False)

        # Strict behavioral proof: The output MUST follow the 'x + 1' branch,
        # proving that is_tracing() returned True and the path was successfully captured.
        traced_output = traced_func(inp)
        self.assertEqual(traced_output, inp + 1,
                         msg="Traced model did not record the is_tracing()==True branch.")


if __name__ == "__main__":
    run_tests()
