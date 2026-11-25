# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
import os
import re
import unittest
from contextlib import ExitStack
from unittest import mock

from packaging.version import Version

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
from torch._inductor import config
from torch._inductor.graph import GraphLowering
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch_npu._inductor.triton_experimental import lowering, lowering_fallback_list as policy

aten = torch.ops.aten
prims = torch.ops.prims
TORCH_216 = Version(torch.__version__).release[:2] == (2, 16)
TRITON_CODEGEN_SKIP_REASON = "Temporarily disabled pending TritonEx codegen validation on PyTorch 2.16"


class TestLoweringFallbackBlacklist(TestCase):
    @staticmethod
    def _triton_kernel_count(body):
        compiled = body.count("async_compile.triton(")
        inlined = len(re.findall(r"(?m)^@npu_triton_heuristics\.\w+\(", body))
        return max(compiled, inlined)

    def test_policy(self):
        for expected, ops in (
            (False, (aten.logical_and, aten.logical_and.default, aten.logical_and.out,
                     torch.ops._c10d_functional.all_reduce.default,
                     torch.ops._c10d_functional.all_reduce_.default,
                     torch.ops._c10d_functional.reduce_scatter_tensor_coalesced.default,
                     torch.ops._c10d_functional.wait_tensor.default,
                     aten.hypot.default, aten.contiguous.default, aten.mul_.Scalar)),
            (True, (aten.embedding.default, aten.cumprod.default, aten.frexp.Tensor,
                    aten.frexp.Tensor_out, aten.bitwise_left_shift.Tensor, prims.bitwise_xor.default,
                    prims.inductor_seed.default, prims.inductor_seeds.default,
                    prims.inductor_random.default, aten.add_.Tensor, aten.mul_.Tensor,
                    aten.logical_or.default, aten.logical_xor.default,
                    aten.randn_like.default, aten._unsafe_index_put.default)),
        ):
            for op in ops:
                with self.subTest(op=op):
                    self.assertEqual(policy.is_lowering_fallback(op), expected)

    def test_registration(self):
        from torch._inductor.fx_passes.control_dependencies import control_deps

        packet, explicit = aten.logical_and, torch.ops.npu._npu_dropout.default
        cases = [
            ((packet,), {str(packet.default)}, {}, (), [packet.default]),
            ((packet, packet.default), {str(packet.default)}, {}, (), [packet.default]),
            ((packet, packet.default), {str(packet)}, {}, (), [packet, packet.default]),
            ((packet, packet.default), {str(packet.default)}, {packet.default: object()}, (), []),
            ((aten.hypot, aten.hypot.default), set(), {}, (), []),
            ((), set(), {}, (explicit,), [explicit]),
            ((control_deps,), {f"higher_order.{control_deps}"}, {}, (), [control_deps]),
        ]
        for keys, names, decompositions, explicit_ops, expected in cases:
            with self.subTest(keys=keys, names=names), ExitStack() as stack:
                original = dict.fromkeys(keys, object())
                registered, recorded, calls = original.copy(), [], mock.Mock()
                patches = dict(lowerings=registered, decompositions=decompositions,
                               EXPLICIT_FALLBACK_LIST=explicit_ops, FALLBACK_LIST=recorded,
                               make_fallback=calls, overwrite_lowering=mock.Mock())
                for name, value in patches.items():
                    stack.enter_context(mock.patch.object(lowering, name, value))
                stack.enter_context(mock.patch.object(policy, "LOWERING_FALLBACK_NAMES", names))
                lowering._register_npu_inductor_fallbacks()
                calls.assert_has_calls([mock.call(op) for op in expected])
                self.assertEqual(calls.call_count, len(expected))
                self.assertEqual(recorded, expected)
                self.assertEqual(registered, original)

    def test_precision_guard(self):
        op, blocked = prims.acos.default, aten.add_.Tensor
        upstream, native, existing = mock.Mock(), mock.Mock(), mock.Mock()
        registered = {op: upstream, blocked: existing}
        with (mock.patch.object(lowering, "lowerings", registered),
              mock.patch.object(lowering, "fallback_handler", return_value=native) as fallback):
            lowering._register_precision_fallbacks()
        fallback.assert_called_once_with(op, add_to_fallback_set=False)
        self.assertIs(registered[blocked], existing)
        narrow = mock.Mock(get_dtype=mock.Mock(return_value=torch.float32))
        registered[op]([narrow])
        upstream.assert_called_once_with([narrow])
        native.assert_not_called()
        for dtype in (torch.int64, torch.float64, torch.complex64, torch.complex128):
            with self.subTest(dtype=dtype):
                wide = mock.Mock(get_dtype=mock.Mock(return_value=dtype))
                registered[op]([narrow, {"value": wide}])
                native.assert_called_with([narrow, {"value": wide}])
                registered[op](narrow, dtype=dtype)
                native.assert_called_with(narrow, dtype=dtype)
        registered[op](narrow, alpha=1j)
        native.assert_called_with(narrow, alpha=1j)
        self.assertEqual(upstream.call_count, 1)

    def _compile(self, fn, args):
        source = []
        torch._dynamo.reset()
        compiled = torch.compile(fn, fullgraph=True, options={"npu_backend": "triton_experimental"})
        with (config.patch("force_disable_caches", True),
              config.patch("implicit_fallbacks", False),
              mock.patch.object(GraphLowering, "save_output_code", source.append)):
            actual = compiled(*args)
        torch.npu.synchronize()
        body = "\n".join(line for line in "\n".join(source).splitlines()
                         if not line.lstrip().startswith("#"))
        return actual, body

    @unittest.skipIf(TORCH_216, TRITON_CODEGEN_SKIP_REASON)
    @unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
    def test_logical_and_codegen_and_fusion(self):
        def fused(x, y):
            return torch.where(aten.logical_and.default(x > 0, y < 1), x, y)

        args = (torch.randn(257, 513, device="npu"), torch.randn(257, 513, device="npu"))
        for fn in (aten.logical_and.default, fused):
            with self.subTest(fn=fn):
                actual, body = self._compile(fn, args)
                torch.testing.assert_close(actual, fn(*args), rtol=0, atol=0)
                self.assertNotIn("torch.ops.aten.logical_and.", body)
                self.assertEqual(self._triton_kernel_count(body), 1)
                self.assertEqual(body.count(".run("), 1)
                if fn is fused:
                    self.assertRegex(body, r"tmp\d+ & tmp\d+")
                    self.assertIn("tl.where(", body)

    @unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
    def test_native_precision(self):
        integers = torch.tensor([0, 1, -1, 2**31, 2**32, 2**40, -(2**40), 2**62], dtype=torch.int64)
        tiny = torch.tensor([0, 1e-300, -1e-300, 1e300, -1e300], dtype=torch.float64)
        cases = [(op, (x, torch.zeros_like(x))) for x in (integers, tiny)
                 for op in (aten.logical_or.default, aten.logical_xor.default)]
        cases += [(op, (integers,)) for op in (aten.sign.default, aten.signbit.default)]
        cases.append((prims.add.default, (integers, torch.ones_like(integers))))
        for op, host_args in cases:
            with self.subTest(op=op, dtype=host_args[0].dtype):
                expected = op(*host_args)
                actual, body = self._compile(op, tuple(t.npu() for t in host_args))
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                self.assertIn(f"torch.ops.{op}(", body)
                self.assertEqual(self._triton_kernel_count(body), 0)

    @unittest.skipIf(TORCH_216, TRITON_CODEGEN_SKIP_REASON)
    @unittest.skipUnless(torch.npu.is_available(), "requires an NPU")
    def test_mutation_and_alias(self):
        x, y = torch.randn(17, 31, device="npu"), torch.randn(1, 31, device="npu")
        out = torch.empty_like(x, dtype=torch.bool)
        actual, _ = self._compile(lambda a, b, c: aten.logical_and.out(a, b, out=c), (x, y, out))
        self.assertEqual(actual.data_ptr(), out.data_ptr())
        torch.testing.assert_close(out, torch.logical_and(x, y), rtol=0, atol=0)
        values = [torch.ones_like(x), torch.full((11, 13), 2.0, device="npu")]
        increments = [torch.full_like(t, 0.5) for t in values]
        expected = [t.cpu() + 0.5 for t in values]

        def update(tensors, additions):
            aten._foreach_add_.List(tensors, additions)

        actual, body = self._compile(update, (values, increments))
        self.assertIsNone(actual)
        for actual, reference in zip(values, expected):
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
        self.assertGreater(self._triton_kernel_count(body), 0)


if __name__ == "__main__":
    run_tests()
