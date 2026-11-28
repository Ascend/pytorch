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
import torch_npu  # noqa: F401  (registers the npu device and its kernels)


def assert_npu_tensor(tensor, name):
    if tensor.device.type != "npu":
        raise AssertionError(f"{name} must be on NPU, got {tensor.device}")


def run_dispatcher_special_zeta():
    # fmax 已注册 NPU kernel，不再作为 fallback 载体；改用尚未注册的 torch.special.zeta
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="npu")
    y = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float32, device="npu")
    expected = torch.special.zeta(x.cpu(), y.cpu())
    for _ in range(2):
        out = torch.special.zeta(x, y)
        assert_npu_tensor(out, "special.zeta result")
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


def run_jagged_nested_ops():
    # to_padded_tensor on a jagged NestedTensor used to fall back to the CPU
    # through aten::_jagged_to_padded_dense_forward.
    lengths = (5, 9, 3)
    offsets = torch.tensor([0, 5, 14, 17], dtype=torch.int64)
    for op_name in ("aten::_jagged_to_padded_dense_forward", "aten::_padded_dense_to_jagged_forward"):
        if not torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "PrivateUse1"):
            raise AssertionError(f"{op_name} has no NPU kernel and would fall back to the CPU")
    nested = torch.nested.as_nested_tensor(
        [torch.randn(length, 3) for length in lengths], layout=torch.jagged
    )
    # CPU references are computed before the target ops, on CPU tensors only.
    expected_padded = torch.nested.to_padded_tensor(nested, padding=0.0)
    expected_values = nested.values().clone()
    expected_dense = torch.ops.aten._jagged_to_padded_dense_forward(expected_values, [offsets], [9], 0.0)

    nested_npu = nested.to("npu:0")
    for _ in range(2):
        padded = torch.nested.to_padded_tensor(nested_npu, padding=0.0)
        assert_npu_tensor(padded, "to_padded_tensor result")
        torch.testing.assert_close(padded.cpu(), expected_padded)

    values_npu = expected_values.npu()
    offsets_npu = offsets.npu()
    dense = torch.ops.aten._jagged_to_padded_dense_forward(values_npu, [offsets_npu], [9], 0.0)
    assert_npu_tensor(dense, "jagged_to_padded_dense result")
    torch.testing.assert_close(dense.cpu(), expected_dense)
    jagged = torch.ops.aten._padded_dense_to_jagged_forward(dense, [offsets_npu], 17)
    assert_npu_tensor(jagged, "padded_dense_to_jagged result")
    torch.testing.assert_close(jagged.cpu(), expected_values)

    # sequences longer than max_lengths are truncated, and bool data must not be
    # silently routed through the CUDA half workaround or the CPU fallback
    truncated = torch.ops.aten._jagged_to_padded_dense_forward(values_npu, [offsets_npu], [4], -1.0)
    assert_npu_tensor(truncated, "truncated jagged_to_padded_dense result")
    torch.testing.assert_close(
        truncated.cpu(),
        torch.ops.aten._jagged_to_padded_dense_forward(expected_values, [offsets], [4], -1.0),
    )
    bool_values = expected_values.gt(0)
    bool_padded = torch.ops.aten._jagged_to_padded_dense_forward(
        bool_values.npu(), [offsets_npu], [9], 0.0)
    assert_npu_tensor(bool_padded, "bool jagged_to_padded_dense result")
    torch.testing.assert_close(
        bool_padded.cpu(),
        torch.ops.aten._jagged_to_padded_dense_forward(bool_values, [offsets], [9], 0.0),
    )

    # backward of to_padded_tensor goes through the padded -> jagged kernel
    grad_values = torch.randn(17, 3, device="npu:0", requires_grad=True)
    grad_nested = torch.nested.nested_tensor_from_jagged(grad_values, offsets_npu, max_seqlen=9)
    padded_grad = torch.nested.to_padded_tensor(grad_nested, padding=0.0)
    grad = torch.randn_like(padded_grad)
    padded_grad.backward(grad)
    assert_npu_tensor(grad_values.grad, "to_padded_tensor grad")
    expected_grad = torch.cat([grad.cpu()[index, :length] for index, length in enumerate(lengths)])
    torch.testing.assert_close(grad_values.grad.cpu(), expected_grad)


torch.npu.set_device(0)
case = sys.argv[1]
if case == "dispatcher_special_zeta":
    run_dispatcher_special_zeta()
elif case == "sparse_csr_sum":
    run_sparse_csr("sum")
elif case == "sparse_csr_prod":
    run_sparse_csr("prod")
elif case == "jagged_nested_ops":
    run_jagged_nested_ops()
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
                    "dispatcher_special_zeta",
                    env_value,
                    "will fall back to run on the CPU",
                    "aten::special_zeta",
                )

    def test_dispatcher_fallback_disabled(self):
        self._assert_fallback_blocked(
            "dispatcher_special_zeta",
            "will fall back to run on the CPU",
            "aten::special_zeta",
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

    def test_jagged_nested_ops_do_not_fall_back(self):
        # Runs jagged to_padded_tensor, the two underlying packing ops and the
        # backward pass in strict mode: any CPU fallback would make the child
        # process fail. The default environment must stay warning-free too.
        for env_value in (None, "1"):
            with self.subTest(env_value=env_value):
                result = self._run_case("jagged_nested_ops", env_value)
                output = self._output(result)
                self.assertEqual(
                    result.returncode,
                    0,
                    f"jagged nested ops must run on NPU, env={env_value!r}\n{output}",
                )
                self.assertIn("CASE_SUCCESS", output)
                self.assertNotIn("will fall back", output.lower())


if __name__ == "__main__":
    run_tests()
