# Copyright (c) 2026 Huawei Technologies Co., Ltd
# Licensed under the BSD 3-Clause License.

import os
import subprocess
import sys
import unittest

import torch

from torch_npu.testing.testcase import TestCase, run_tests


CPU_FALLBACK_ENV = "TORCH_NPU_FALLBACK_CPU_DISABLE"
SUBPROCESS_TIMEOUT = 120


# OptionsManager caches the environment variable in a function-local static.
# Every case must therefore run in a fresh process, with the environment set
# before torch_npu is imported.
_CHILD_SCRIPT = r"""
import sys

import torch


def assert_npu_tensor(tensor, name):
    if tensor.device.type != "npu":
        raise AssertionError(f"{name} must be on NPU, got {tensor.device}")


def run_dispatcher_fmax_out():
    x = torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float32, device="npu")
    y = torch.tensor([2.0, 4.0, float("nan")], dtype=torch.float32, device="npu")
    expected = torch.tensor([2.0, 4.0, 3.0], dtype=torch.float32)
    for _ in range(2):
        out = torch.empty_like(x)
        torch.fmax(x, y, out=out)
        assert_npu_tensor(out, "fmax.out result")
        torch.testing.assert_close(out.cpu(), expected)


def make_sparse_csr():
    crow = torch.tensor([0, 2, 4], dtype=torch.int64, device="npu")
    col = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="npu")
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="npu")
    return torch.sparse_csr_tensor(crow, col, values, size=(2, 2))


def run_sparse_csr(reduction):
    value = make_sparse_csr()
    op = getattr(torch.ops.aten, f"_sparse_csr_{reduction}").dim_dtype
    for _ in range(2):
        result = op(value, [1], True, dtype=None)
        assert_npu_tensor(result, f"sparse_csr_{reduction} result")
        # Materialize the result to ensure the fallback and NPU copy-back have
        # completed. No CPU reference reduction is run before the target op.
        result.cpu()


def run_normal_npu():
    x = torch.arange(8, dtype=torch.float32, device="npu")
    y = torch.ones(8, dtype=torch.float32, device="npu")
    expected = torch.arange(8, dtype=torch.float32) + 1
    for _ in range(2):
        result = torch.add(x, y)
        assert_npu_tensor(result, "add result")
        torch.testing.assert_close(result.cpu(), expected)


torch.npu.set_device(0)
case = sys.argv[1]
if case == "dispatcher_fmax_out":
    run_dispatcher_fmax_out()
elif case == "sparse_csr_sum":
    run_sparse_csr("sum")
elif case == "sparse_csr_prod":
    run_sparse_csr("prod")
elif case == "normal_npu":
    run_normal_npu()
else:
    raise AssertionError(f"unknown case: {case}")

torch.npu.synchronize()
print("CASE_SUCCESS")
"""


@unittest.skipIf(not torch.npu.is_available(), "requires NPU")
class TestCpuFallbackControl(TestCase):
    @staticmethod
    def _run_case(case, env_value):
        env = os.environ.copy()
        if env_value is None:
            env.pop(CPU_FALLBACK_ENV, None)
        else:
            env[CPU_FALLBACK_ENV] = env_value
        return subprocess.run(
            [sys.executable, "-c", _CHILD_SCRIPT, case],
            env=env,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
            check=False,
        )

    @staticmethod
    def _output(result):
        return (result.stdout or "") + (result.stderr or "")

    def _assert_fallback_allowed(self, case, env_value, warning_needle, op_name=None):
        result = self._run_case(case, env_value)
        output = self._output(result)
        self.assertEqual(
            result.returncode,
            0,
            f"fallback should be allowed for {case}, env={env_value!r}\n{output}",
        )
        self.assertIn("CASE_SUCCESS", output)
        self.assertEqual(
            output.lower().count(warning_needle.lower()),
            1,
            f"fallback warning must be emitted once for {case}\n{output}",
        )
        if op_name is not None:
            self.assertIn(op_name, output)

    def _assert_fallback_blocked(self, case, warning_needle, op_name=None):
        result = self._run_case(case, "1")
        output = self._output(result)
        self.assertNotEqual(
            result.returncode,
            0,
            f"fallback should be blocked for {case}\n{output}",
        )
        self.assertNotIn("CASE_SUCCESS", output)
        self.assertIn(CPU_FALLBACK_ENV, output)
        self.assertNotIn(
            warning_needle.lower(),
            output.lower(),
            f"strict mode must fail before the fallback warning for {case}\n{output}",
        )
        if op_name is not None:
            self.assertIn(op_name, output)

    def test_dispatcher_fallback_default_and_explicitly_allowed(self):
        for env_value in (None, "0"):
            with self.subTest(env_value=env_value):
                self._assert_fallback_allowed(
                    "dispatcher_fmax_out",
                    env_value,
                    "will fall back to run on the CPU",
                    "aten::fmax.out",
                )

    def test_dispatcher_fallback_disabled(self):
        self._assert_fallback_blocked(
            "dispatcher_fmax_out",
            "will fall back to run on the CPU",
            "aten::fmax.out",
        )

    def test_sparse_csr_sum_fallback_default_and_explicitly_allowed(self):
        for env_value in (None, "0"):
            with self.subTest(env_value=env_value):
                self._assert_fallback_allowed(
                    "sparse_csr_sum",
                    env_value,
                    "will fall back to CPU",
                    "aten::_sparse_csr_sum.dim_dtype",
                )

    def test_sparse_csr_sum_fallback_disabled(self):
        self._assert_fallback_blocked(
            "sparse_csr_sum",
            "will fall back to CPU",
            "aten::_sparse_csr_sum.dim_dtype",
        )

    def test_sparse_csr_prod_fallback_default_and_explicitly_allowed(self):
        for env_value in (None, "0"):
            with self.subTest(env_value=env_value):
                self._assert_fallback_allowed(
                    "sparse_csr_prod",
                    env_value,
                    "will fall back to CPU",
                    "aten::_sparse_csr_prod.dim_dtype",
                )

    def test_sparse_csr_prod_fallback_disabled(self):
        self._assert_fallback_blocked(
            "sparse_csr_prod",
            "will fall back to CPU",
            "aten::_sparse_csr_prod.dim_dtype",
        )

    def test_normal_npu_kernel_is_not_blocked(self):
        result = self._run_case("normal_npu", "1")
        output = self._output(result)
        self.assertEqual(
            result.returncode,
            0,
            f"strict mode must not block a normal NPU kernel\n{output}",
        )
        self.assertIn("CASE_SUCCESS", output)
        self.assertNotIn("will fall back", output.lower())
        self.assertNotIn("fallback to run on the cpu", output.lower())


if __name__ == "__main__":
    run_tests()
