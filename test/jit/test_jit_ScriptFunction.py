# Owner(s): ["oncall: jit"]
"""
add validation cases for torch.jit.ScriptFunction APIs:
torch.jit.ScriptFunction
torch.jit.ScriptFunction.get_debug_state
torch.jit.ScriptFunction.save
torch.jit.ScriptFunction.save_to_buffer
"""

import io
import tempfile
import os
import torch
import unittest
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
    enable_profiling_mode_for_profiling_tests,
    GRAPH_EXECUTOR,
    ProfilingMode,
)


class TestScriptFunctionAPI(TestCase):
    def setUp(self):
        super().setUp()

        @torch.jit.script
        def test_fn(x: torch.Tensor) -> torch.Tensor:
            return x * 2 + 1

        self.script_fn = test_fn
        self.dummy_input = torch.tensor([1.0, 2.0, 3.0]).npu()

    def test_script_function_type(self):
        """Verify the object is a torch.jit.ScriptFunction."""
        self.assertIsInstance(self.script_fn, torch.jit.ScriptFunction)

    @unittest.skipIf(
        GRAPH_EXECUTOR != ProfilingMode.PROFILING,
        "get_debug_state requires profiling graph executor",
    )
    def test_get_debug_state(self):
        """Verify get_debug_state returns a valid object under profiling mode."""
        with enable_profiling_mode_for_profiling_tests():
            # Run twice: first to profile, second to generate optimized plan
            self.script_fn(self.dummy_input)
            self.script_fn(self.dummy_input)
            state = self.script_fn.get_debug_state()
            self.assertIsNotNone(state)
            self.assertTrue(hasattr(state, "execution_plans"))

    def test_save_to_buffer(self):
        """Test save_to_buffer returns bytes and can be loaded back."""
        buf = self.script_fn.save_to_buffer()
        self.assertIsInstance(buf, bytes)
        loaded = torch.jit.load(io.BytesIO(buf))
        expected = self.script_fn(self.dummy_input)
        actual = loaded(self.dummy_input)
        self.assertTrue(torch.equal(expected, actual))

    def test_save_to_file(self):
        """Test save with a file path and reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "script_fn.pt")
            self.script_fn.save(fname)
            loaded = torch.jit.load(fname)
            expected = self.script_fn(self.dummy_input)
            actual = loaded(self.dummy_input)
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    run_tests()
