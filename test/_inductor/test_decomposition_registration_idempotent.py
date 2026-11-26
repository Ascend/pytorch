# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Re-entering the Triton backend must not re-register decompositions (#5681).

``_load_backend`` runs the per-backend registrars, and it is entered more than
once per process (backend switch, a second ``register_inductor_npu``). A
registrar that registers an op already present in ``decompositions`` makes
``torch._decomp._add_op_to_registry`` raise
``RuntimeError: duplicate registrations for aten.erfc.default``.

These cases pin the properties that keep that from coming back:

1. Calling the registrar twice raises nothing.
2. Re-registering leaves every decomposition this registrar does not own
   untouched, and does not add or drop entries.
3. A second ``_load_backend`` for the same backend does not re-run the loader,
   while a backend *switch* still does. This is what makes a new
   ``register_decomposition`` safe by construction: without it, every new
   registration would again need a hand-kept cleanup list to stay re-entrant --
   exactly the list ``aten.erfc`` was missing from.
"""

from unittest import mock

import torch
from torch._inductor.decomposition import decompositions, fast_random_decomps
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu  # noqa: F401
import torch_npu._inductor as npu_inductor
import torch_npu._inductor.lowering_patch as lowering_patch
from torch_npu._inductor.decomposition import _register_triton_decompositions
from torch_npu.utils._dynamo import _InductorNpuRegistry

aten = torch.ops.aten

# Ops the Triton registrar registers unconditionally. ``aten.erfc`` is the one
# that used to be missing from the cleanup list and started #5681.
# ``aten.expm1`` is registered by a different registrar (see the bare
# ``register_decomposition(torch.ops.aten.expm1)`` near the end of the module),
# so it is neither owned nor re-registered here.
_REGISTERED_OPS = [
    aten.erfc.default,
    aten.matmul.default,
    aten.matmul_backward.default,
]

# The registrar takes OpOverloadPackets, so every overload of those packets is
# (re)registered -- ``aten.erfc.out`` and friends included.
_OWNED_OPS = {
    getattr(packet, overload)
    for packet in (
        aten.erfc,
        aten.matmul,
        aten.matmul_backward,
    )
    for overload in packet.overloads()
}


class TestDecompositionRegistrationIdempotent(TestCase):
    def test_register_twice_does_not_raise(self):
        _register_triton_decompositions()
        # The historical failure: the second entry raised
        # `RuntimeError: duplicate registrations for aten.erfc.default`.
        _register_triton_decompositions()

        for op in _REGISTERED_OPS:
            self.assertIn(op, decompositions)

    def test_second_register_keeps_table_shape(self):
        _register_triton_decompositions()
        before = set(decompositions)
        _register_triton_decompositions()
        # Re-entry must not add or drop entries. (The registered functions are
        # closures rebuilt per call, so their identity changes on every call --
        # the op set is the meaningful invariant, not the function objects.)
        self.assertEqual(before, set(decompositions))

    def test_second_register_leaves_foreign_decompositions_alone(self):
        _register_triton_decompositions()
        foreign = {
            op: fn for op, fn in decompositions.items() if op not in _OWNED_OPS
        }
        self.assertTrue(foreign, "expected upstream decompositions to be present")
        _register_triton_decompositions()

        for op, fn in foreign.items():
            self.assertIs(fn, decompositions.get(op), msg=f"{op} was replaced")

    def test_second_load_backend_skips_loader(self):
        backend = npu_inductor._get_backend()
        with mock.patch.object(_InductorNpuRegistry, "_loaded_backend", backend):
            loader = mock.Mock()
            with mock.patch.dict(
                npu_inductor._BACKEND_LOADERS, {backend: loader}, clear=False
            ):
                npu_inductor._load_backend()

        loader.assert_not_called()
        self.assertEqual(_InductorNpuRegistry._loaded_backend, backend)

    def test_backend_switch_still_runs_loader(self):
        # The guard must key on the backend name, not act as a one-shot.
        backend = npu_inductor._get_backend()
        with mock.patch.object(_InductorNpuRegistry, "_loaded_backend", "some-other"):
            loader = mock.Mock()
            with mock.patch.dict(
                npu_inductor._BACKEND_LOADERS, {backend: loader}, clear=False
            ), mock.patch.object(
                lowering_patch, "restore_inductor_baseline", mock.Mock()
            ), mock.patch.object(
                npu_inductor, "_apply_npu_cpp_march_default", mock.Mock()
            ), mock.patch(
                "torch_npu._inductor.fx_passes.overlap_scheduling.patch_overlap_scheduling",
                mock.Mock(),
            ), mock.patch.object(
                fast_random_decomps, "cache_clear", mock.Mock()
            ):
                npu_inductor._load_backend()

        loader.assert_called_once_with()
        self.assertEqual(_InductorNpuRegistry._loaded_backend, backend)


if __name__ == "__main__":
    run_tests()
