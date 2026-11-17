import os
import subprocess
import sys
import textwrap

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)

import torch_npu  # noqa: F401
from torch_npu.utils._dynamo import register_inductor_npu


register_inductor_npu()


class TestExplicitFallbackList(TestCase):
    """Every EXPLICIT_FALLBACK_LIST entry must end up with a fallback lowering.

    The bulk registration pass walks the upstream ``lowerings`` table, so an op
    with no upstream lowering is only reachable through this list. Missing one
    makes a strict compile (``implicit_fallbacks=False``) raise
    ``MissingOperatorWithoutDecomp`` / ``MissingOperatorWithDecomp``.
    """

    def _assert_registered(self, op):
        from torch._inductor.lowering import lowerings

        self.assertIn(op, lowerings)

    def test_every_explicit_entry_is_registered(self):
        from torch._inductor.lowering import lowerings
        from torch_npu._inductor.triton_experimental.lowering_fallback_list import (
            EXPLICIT_FALLBACK_LIST,
        )

        missing = [str(op) for op in EXPLICIT_FALLBACK_LIST if op not in lowerings]
        self.assertEqual(missing, [])

    @parametrize('op_path', [
        'aten.histogram.bins_tensor',
        'aten._histogramdd_from_bin_tensors.default',
        'aten.replication_pad3d_backward.default',
        'npu.repeat_interleave_backward_int.default',
    ])
    def test_added_ops_are_registered(self, op_path):
        namespace, packet, overload = op_path.split('.')
        op = getattr(getattr(getattr(torch.ops, namespace), packet), overload)
        self._assert_registered(op)

    def test_embedding_backward_registration_matches_device(self):
        from torch._inductor.decomposition import decompositions
        from torch_npu._inductor.triton_experimental import device_props
        from torch_npu._inductor.triton_experimental.lowering import FALLBACK_LIST

        op = torch.ops.aten.embedding_dense_backward.default
        if device_props.is_a5():
            self.assertIn(op, decompositions)
            self.assertNotIn(op, FALLBACK_LIST)
        else:
            self.assertNotIn(op, decompositions)
            self.assertIn(op, FALLBACK_LIST)
            self._assert_registered(op)

    def test_a5_backend_initialization_preserves_embedding_decomposition(self):
        # Simulate only the capability probe; no A5 kernels execute here.
        code = textwrap.dedent("""
            import torch
            import torch_npu
            from unittest import mock
            from torch_npu.utils._dynamo import register_inductor_npu

            with mock.patch.object(torch_npu._C, '_npu_get_soc_version', return_value=250):
                register_inductor_npu()

            from torch._inductor.decomposition import decompositions
            from torch_npu._inductor.triton_experimental import device_props
            from torch_npu._inductor.triton_experimental.lowering import FALLBACK_LIST
            from torch_npu._inductor.triton_experimental.lowering_fallback_list import (
                EXPLICIT_FALLBACK_LIST,
            )

            op = torch.ops.aten.embedding_dense_backward.default
            assert device_props.is_a5()
            assert op in decompositions
            assert op not in EXPLICIT_FALLBACK_LIST
            assert op not in FALLBACK_LIST
        """)
        env = dict(os.environ, TORCHINDUCTOR_NPU_BACKEND="triton_experimental")
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            timeout=180, env=env,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_embedding_backward_compiles_and_matches_eager(self):
        weight = torch.randn(10, 4, device='npu', requires_grad=True)
        indices = torch.tensor([1, 3, 3, 7], device='npu')

        def fn(w, idx):
            # F.embedding takes (indices, weight).
            return F.embedding(idx, w).sum()

        fn(weight, indices).backward()
        eager_grad = weight.grad.clone()
        weight.grad = None

        torch.compile(fn, backend="inductor")(weight, indices).backward()

        self.assertEqual(weight.grad.dtype, eager_grad.dtype)
        self.assertEqual(weight.grad.shape, eager_grad.shape)
        self.assertEqual(weight.grad, eager_grad)

    def test_replication_pad3d_backward_compiles_and_matches_eager(self):
        x = torch.randn(1, 2, 3, 4, 5, device='npu', requires_grad=True)

        def fn(t):
            return F.pad(t, (1, 1, 1, 1, 1, 1), mode='replicate').sum()

        fn(x).backward()
        eager_grad = x.grad.clone()
        x.grad = None

        torch.compile(fn, backend="inductor")(x).backward()

        self.assertEqual(x.grad.dtype, eager_grad.dtype)
        self.assertEqual(x.grad.shape, eager_grad.shape)
        self.assertEqual(x.grad, eager_grad)

    def test_repeat_interleave_backward_compiles_and_matches_eager(self):
        x = torch.randn(3, 4, device='npu', requires_grad=True)

        def fn(t):
            return torch.repeat_interleave(t, 2, dim=0).sum()

        fn(x).backward()
        eager_grad = x.grad.clone()
        x.grad = None

        torch.compile(fn, backend="inductor")(x).backward()

        self.assertEqual(x.grad.dtype, eager_grad.dtype)
        self.assertEqual(x.grad.shape, eager_grad.shape)
        self.assertEqual(x.grad, eager_grad)


instantiate_parametrized_tests(TestExplicitFallbackList)

if __name__ == "__main__":
    run_tests()
