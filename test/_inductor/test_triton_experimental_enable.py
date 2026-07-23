# Owner(s): ["module: tests"]
import os

import torch
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils

import torch_npu  # noqa: F401

# Heuristics import emitted only by the triton_experimental wrapper header
# (torch_npu/_inductor/triton_experimental/codegen/wrapper.py:74). The default
# Triton backend instead emits torch_npu._inductor.runtime.triton_heuristics, so
# this substring cleanly identifies which codegen backend produced the wrapper.
# Deliberately the heuristics import (not a bare "triton_experimental" match):
# the get_raw_stream device-op override is a process-global last-writer-wins
# registration, so a plain package-name match would false-positive on default
# once triton_experimental has been activated earlier in the same process.
EXPERIMENTAL_MARKER = "triton_experimental import npu_triton_heuristics"
# Heuristics import emitted only by the default Triton wrapper header
# (torch_npu/_inductor/codegen/wrapper.py:202).
DEFAULT_MARKER = "torch_npu._inductor.runtime.triton_heuristics as triton_heuristics"


class TestTritonExperimentalEnable(TestUtils):

    def setUp(self):
        super().setUp()
        # Reset dynamo/inductor caches so each entry-point test forces codegen
        # with the backend it selects (otherwise a cached artifact from another
        # backend could be reused and run_and_get_code would return stale code).
        torch._dynamo.reset()
        self._saved_config = getattr(torch._inductor.config, "npu_backend", "default")
        self._saved_env = os.environ.get("TORCHINDUCTOR_NPU_BACKEND")

    def tearDown(self):
        # Restore the global backend selectors so backends stay isolated and the
        # next test file does not inherit triton_experimental.
        torch._inductor.config.npu_backend = self._saved_config
        if self._saved_env is None:
            os.environ.pop("TORCHINDUCTOR_NPU_BACKEND", None)
        else:
            os.environ["TORCHINDUCTOR_NPU_BACKEND"] = self._saved_env
        torch._dynamo.reset()
        super().tearDown()

    def op_calc(self, first_element, second_element):
        return first_element + second_element

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float32', 'int64'])
    def test_options_entry(self, shape, dtype):
        # Entry point 1: torch.compile(options={"npu_backend": "triton_experimental"}).
        x = self._generate_tensor(shape, dtype)
        y = self._generate_tensor(shape, dtype)
        std_out = self.op_calc(x, y)
        compile_func = torch.compile(
            self.op_calc, options={"npu_backend": "triton_experimental"}
        )
        compile_out, codes = run_and_get_code(compile_func, x, y)
        self.assertEqual(std_out, compile_out)
        self.assertIn(EXPERIMENTAL_MARKER, codes[0])

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float32', 'int64'])
    def test_config_entry(self, shape, dtype):
        # Entry point 2: global torch._inductor.config.npu_backend.
        torch._inductor.config.npu_backend = "triton_experimental"
        x = self._generate_tensor(shape, dtype)
        y = self._generate_tensor(shape, dtype)
        std_out = self.op_calc(x, y)
        compile_func = torch.compile(self.op_calc)
        compile_out, codes = run_and_get_code(compile_func, x, y)
        self.assertEqual(std_out, compile_out)
        self.assertIn(EXPERIMENTAL_MARKER, codes[0])

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float32', 'int64'])
    def test_env_entry(self, shape, dtype):
        # Entry point 3: TORCHINDUCTOR_NPU_BACKEND environment variable.
        os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "triton_experimental"
        x = self._generate_tensor(shape, dtype)
        y = self._generate_tensor(shape, dtype)
        std_out = self.op_calc(x, y)
        compile_func = torch.compile(self.op_calc)
        compile_out, codes = run_and_get_code(compile_func, x, y)
        self.assertEqual(std_out, compile_out)
        self.assertIn(EXPERIMENTAL_MARKER, codes[0])


class TestTritonExperimentalIsolation(TestUtils):
    # Validates the design decision (2026-07-13) that triton_experimental and the
    # default Triton backend are physically isolated: both must compile correctly
    # within one process/scope, each emitting its own wrapper header.

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self._saved_config = getattr(torch._inductor.config, "npu_backend", "default")
        self._saved_env = os.environ.get("TORCHINDUCTOR_NPU_BACKEND")

    def tearDown(self):
        torch._inductor.config.npu_backend = self._saved_config
        if self._saved_env is None:
            os.environ.pop("TORCHINDUCTOR_NPU_BACKEND", None)
        else:
            os.environ["TORCHINDUCTOR_NPU_BACKEND"] = self._saved_env
        torch._dynamo.reset()
        super().tearDown()

    @parametrize('dtype', ['float32'])
    def test_default_and_experimental_in_one_scope(self, dtype):
        shape = (1024, 32)
        a = self._generate_tensor(shape, dtype)
        b = self._generate_tensor(shape, dtype)
        x = self._generate_tensor(shape, dtype)
        y = self._generate_tensor(shape, dtype)

        # Distinct functions so the two compiles never share a dynamo cache entry.
        @torch.compile()
        def op_default(x, y):
            return x + y

        default_out, default_codes = run_and_get_code(op_default, x, y)
        self.assertEqual(x + y, default_out)
        # Default wrapper carries the default heuristics import and none from
        # the experimental codegen backend.
        self.assertIn(DEFAULT_MARKER, default_codes[0])
        self.assertNotIn(EXPERIMENTAL_MARKER, default_codes[0])

        @torch.compile(options={"npu_backend": "triton_experimental"})
        def op_experimental(a, b):
            return a - b

        exp_out, exp_codes = run_and_get_code(op_experimental, a, b)
        self.assertEqual(a - b, exp_out)
        self.assertIn(EXPERIMENTAL_MARKER, exp_codes[0])


instantiate_parametrized_tests(TestTritonExperimentalEnable)
instantiate_parametrized_tests(TestTritonExperimentalIsolation)

if __name__ == "__main__":
    run_tests()
