import unittest

import torch
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TestCase,
)

import torch_npu


if not torch_npu.npu.is_available():
    raise unittest.SkipTest("NPU is not available")

device = "npu"


class TestSymbolicGroupElementwise(TestCase):
    def setUp(self):
        super().setUp()
        import torch_npu._inductor.config as npu_config

        self.npu_config = npu_config
        self.prev_group_autotune = npu_config.enable_symbolic_shape_group_autotune
        npu_config.enable_symbolic_shape_group_autotune = True
        torch._dynamo.reset()

    def tearDown(self):
        self.npu_config.enable_symbolic_shape_group_autotune = self.prev_group_autotune
        torch._dynamo.reset()
        super().tearDown()

    def _run_and_check(self, fn, inputs, next_inputs, expected_workload):
        for current, next_value in zip(inputs, next_inputs):
            if (
                isinstance(current, torch.Tensor)
                and current.ndim
                and current.shape[0] != next_value.shape[0]
            ):
                torch._dynamo.mark_dynamic(current, 0)

        expected = fn(*inputs)
        compiled = torch.compile(fn, backend="inductor")
        actual, codes = run_and_get_code(compiled, *inputs)
        torch.testing.assert_close(actual, expected)

        next_expected = fn(*next_inputs)
        next_actual = compiled(*next_inputs)
        torch.testing.assert_close(next_actual, next_expected)

        expected_metadata = f"'group_workload': {expected_workload!r}"
        matching_codes = [
            code
            for code in codes
            if "'group_enabled': True" in code
            and "'group_template': 'pointwise'" in code
            and expected_metadata in code
        ]
        self.assertTrue(
            matching_codes,
            f"Expected pointwise group metadata with {expected_metadata}, got:\n{codes}",
        )

    def test_basic_elementwise_workload(self):
        def fn(x, y):
            return torch.relu(x + y) * 0.5

        inputs = (
            torch.randn((257, 1031), device=device),
            torch.randn((257, 1031), device=device),
        )
        next_inputs = (
            torch.randn((263, 1031), device=device),
            torch.randn((263, 1031), device=device),
        )
        self._run_and_check(fn, inputs, next_inputs, "elementwise")

    def test_broadcast_is_not_elementwise_workload(self):
        def fn(x, bias):
            return torch.relu(x + bias) * 0.5

        bias = torch.randn((1031,), device=device)
        inputs = (torch.randn((257, 1031), device=device), bias)
        next_inputs = (torch.randn((263, 1031), device=device), bias)
        self._run_and_check(fn, inputs, next_inputs, None)

    def test_consumed_full_is_elementwise_workload(self):
        def fn(x):
            generated = torch.full_like(x, 2.0)
            return torch.relu(x + generated)

        inputs = (torch.randn((257, 1031), device=device),)
        next_inputs = (torch.randn((263, 1031), device=device),)
        self._run_and_check(fn, inputs, next_inputs, "elementwise")

    def test_standalone_full_is_not_elementwise_workload(self):
        def fn(x):
            return torch.full_like(x, 2.0)

        inputs = (torch.randn((257, 1031), device=device),)
        next_inputs = (torch.randn((263, 1031), device=device),)
        self._run_and_check(fn, inputs, next_inputs, None)

    def test_strided_reindex_is_not_elementwise_workload(self):
        def fn(x, y):
            return torch.relu(x[:, ::2] + y)

        inputs = (
            torch.randn((257, 2062), device=device),
            torch.randn((257, 1031), device=device),
        )
        next_inputs = (
            torch.randn((263, 2062), device=device),
            torch.randn((263, 1031), device=device),
        )
        self._run_and_check(fn, inputs, next_inputs, None)


instantiate_parametrized_tests(TestSymbolicGroupElementwise)


if __name__ == "__main__":
    run_tests()
