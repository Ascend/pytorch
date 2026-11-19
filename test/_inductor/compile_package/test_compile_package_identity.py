# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# lintrunner: skip PYFMT
# Owner(s): ["module: inductor"]
"""Identity and request-key coverage for the DVM compile-package adapters.

The producer identity lives under :mod:`torch_npu._inductor.dvm.persistence`:
the environment fingerprint (``collect_identity``), the codegen configuration
projection (the ``*_FIELDS`` tables), and the UUID the backend pass publishes
into the FX/AOT cache keys
(:class:`~torch_npu._inductor.dvm.persistence.DvmCacheIdentity`). The
backend-neutral format they serialize through is
:mod:`torch_npu.dynamo._package_compat.format`.

None of these tables is enforced anywhere else: a new codegen knob that is
forgotten here silently reuses artifacts compiled under different options, and
a knob that is added needlessly turns every reuse into a miss. The projection
tests below are the guard that turns both mistakes into failures.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch_npu  # noqa: F401  - registers torch.npu
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu._inductor.dvm.persistence import (
    CompileRequest,
    RequestContext,
    capture_configs,
    record_initialization,
)
from torch_npu._inductor.dvm.persistence.identity import (
    ANIR_FIELDS,
    COMMON_FIELDS,
    DVM_FIELDS,
    collect_identity,
    file_digest,
)
from torch_npu.dynamo._package_compat.format import (
    IdentityUnavailable,
    Manifest,
    stable_hash,
)

RUN_NPU = torch.npu.is_available()


def _set_field(module, field, value):
    """Assign through a dotted path, the way the snapshot reads it."""
    parts = field.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def _mutated(value):
    """A different value of the same kind, or None when there is no rule."""
    if value is None:
        return "mutated"
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + "_mutated"
    if isinstance(value, list):
        return [*value, "mutated"]
    if isinstance(value, tuple):
        return (*value, "mutated")
    if isinstance(value, (set, frozenset)):
        return {*value, "mutated"}
    if isinstance(value, dict):
        return {**value, "mutated": 1}
    return None


class TestFileDigest(TestCase):
    """Content identity of the files the compiler installation is keyed on."""

    def test_a_missing_file_is_unreadable(self):
        with self.assertRaises(IdentityUnavailable) as ctx:
            file_digest(Path(tempfile.gettempdir()) / "no-such-file-261111.py")
        self.assertIn("NPU_DEPENDENCY_UNREADABLE", str(ctx.exception))

    def test_a_directory_is_unreadable(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(IdentityUnavailable) as ctx:
                file_digest(Path(folder))
        self.assertIn("NPU_DEPENDENCY_UNREADABLE", str(ctx.exception))

    def test_the_digest_is_stable_and_content_addressed(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "one"
            second = Path(folder) / "two"
            first.write_text("same content")
            second.write_text("same content")
            third = Path(folder) / "three"
            third.write_text("other content")

            digest = file_digest(first)
            self.assertEqual(len(digest), 64)
            self.assertEqual(file_digest(first), digest)
            self.assertEqual(file_digest(second), digest)
            self.assertNotEqual(file_digest(third), digest)


class TestCollectIdentityBranches(TestCase):
    """Every refusal branch of the environment fingerprint."""

    def test_a_non_dvm_backend_is_out_of_scope(self):
        with self.assertRaises(IdentityUnavailable) as ctx:
            collect_identity((torch.device("npu", 0),), "ascendc")
        self.assertIn("NPU_BACKEND_OUT_OF_SCOPE", str(ctx.exception))

    def test_no_target_is_unknown(self):
        with self.assertRaises(IdentityUnavailable) as ctx:
            collect_identity((), "dvm")
        self.assertIn("NPU_TARGET_UNKNOWN", str(ctx.exception))

    def test_multiple_targets_are_unsupported(self):
        # Multi-card requests are refused rather than keyed on one card: an
        # artifact built for a device set is not silently reused elsewhere.
        with self.assertRaises(IdentityUnavailable) as ctx:
            collect_identity((torch.device("npu", 0), torch.device("npu", 1)), "dvm")
        self.assertIn("NPU_MULTIPLE_TARGETS_UNSUPPORTED", str(ctx.exception))

    @unittest.skipIf(not RUN_NPU, "requires npu")
    def test_an_unknown_cann_release_is_refused(self):
        with mock.patch(
            "torch_npu.npu.utils.get_cann_version", return_value="unknown"
        ):
            with self.assertRaises(IdentityUnavailable) as ctx:
                collect_identity((torch.device("npu", 0),), "dvm")
        self.assertIn("NPU_CANN_IDENTITY_UNKNOWN", str(ctx.exception))

    @unittest.skipIf(not RUN_NPU, "requires npu")
    def test_a_live_environment_yields_an_identity(self):
        identity = collect_identity((torch.device("npu", 0),), "dvm")

        self.assertEqual(identity["schema"], 1)
        self.assertTrue(identity["targets"])
        self.assertEqual(
            set(identity["cann"]), {"CANN", "RUNTIME", "COMPILER", "OPP"})
        # The DVM builder modules are part of the fingerprint, so an edit to
        # the codegen itself invalidates artifacts built before it.
        self.assertEqual(
            set(identity["dvm"]),
            {"graph_build", "op_emitter", "mlir_fusion", "config", "decomp"})
        self.assertIn("version", identity["torch_npu"])


class TestConfigProjection(TestCase):
    """The tables that decide which knobs move a cache key.

    For every tracked field: mutate it, capture the projection, and require
    the hash to move. A field with no mutation rule is reported as skipped so
    a new value type cannot quietly fall out of the guard.
    """

    TRACKED = (
        ("torch_npu._inductor.config", COMMON_FIELDS),
        ("torch_npu._inductor.dvm.config", DVM_FIELDS),
        ("torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.config", ANIR_FIELDS),
    )

    def test_every_tracked_field_moves_the_config_hash(self):
        import importlib

        from torch_npu._inductor.dvm.persistence.identity import _getattr_path

        checked, skipped = [], []
        baseline = stable_hash(capture_configs("dvm"))
        for module_name, fields in self.TRACKED:
            module = importlib.import_module(module_name)
            for field in fields:
                original = _getattr_path(module, field)
                replacement = _mutated(original)
                if replacement is None:
                    skipped.append(f"{module_name}.{field}:{type(original).__name__}")
                    continue
                try:
                    _set_field(module, field, replacement)
                    self.assertNotEqual(
                        stable_hash(capture_configs("dvm")), baseline,
                        f"{module_name}.{field} does not move the config hash")
                finally:
                    _set_field(module, field, original)
                checked.append(f"{module_name}.{field}")
        self.assertTrue(checked, "no field could be mutated at all")
        # The projection is only as complete as this guard; list what it
        # could not check so a gap is visible instead of silent.
        self.assertTrue(skipped == [], f"fields without a mutation rule: {skipped}")

    def test_excluded_options_do_not_move_the_request_identity(self):
        from torch_npu._inductor.dvm.persistence.identity import _options_identity

        first = _options_identity(
            {"npu_backend": "dvm", "compile_threads": 1, "trace.depth": 2, "mode": "a"})
        second = _options_identity(
            {"npu_backend": "eager", "compile_threads": 8,
             "trace.other": 3, "mode": "a"})
        self.assertEqual(first, second)

    def test_compiler_options_do_move_the_request_identity(self):
        from torch_npu._inductor.dvm.persistence.identity import _options_identity

        first = _options_identity({"mode": "a"})
        second = _options_identity({"mode": "b"})
        self.assertNotEqual(first, second)

    def test_an_in_place_edit_of_a_request_is_detected(self):
        request = CompileRequest({"mode": "a", "npu_backend": "dvm"})

        request.options["mode"] = "b"
        with self.assertRaises(IdentityUnavailable) as ctx:
            request.check_identity()
        self.assertIn("NPU_CONFIG_IDENTITY_MISMATCH", str(ctx.exception))

    def test_initialization_drift_is_detected(self):
        """Knobs that only take effect at activation must not be replayed."""
        from torch_npu._inductor.dvm.persistence.identity import initialization_state

        record_initialization()
        import torch_npu._inductor.dvm.config as dvm_config

        original = dvm_config.enable_decomp_list
        try:
            dvm_config.enable_decomp_list = [*(original or []), "mutated"]
            with self.assertRaises(IdentityUnavailable) as ctx:
                initialization_state()
        finally:
            dvm_config.enable_decomp_list = original
        self.assertIn("NPU_INITIALIZATION_CONFIG_MISMATCH", str(ctx.exception))


class TestDvmCacheIdentityUuid(TestCase):
    """The UUID every FX/AOT key includes, via the backend custom pass."""

    def _uuid(self, context):
        from torch_npu.dynamo._package_compat.context import ACTIVE
        from torch_npu._inductor.dvm.persistence import DvmCacheIdentity

        token = ACTIVE.set(context)
        try:
            return DvmCacheIdentity().uuid()
        finally:
            ACTIVE.reset(token)

    def test_an_inactive_request_yields_a_neutral_uuid(self):
        # Outside a DVM request the pass must not probe devices or capture
        # ambient configuration: CPU and other-backend graphs key on it.
        self.assertEqual(self._uuid(None), ("torch_npu.dvm", "inactive"))

    def test_an_uncacheable_request_bypasses_the_cache(self):
        from torch._inductor.codecache import BypassFxGraphCache

        context = RequestContext.uncacheable(
            "dvm", (torch.device("npu", 0),), "NPU_REQUEST_UNCACHEABLE")
        with self.assertRaises(BypassFxGraphCache):
            self._uuid(context)

    def test_a_persistable_request_publishes_its_manifest(self):
        manifest = Manifest.create({"backend": "dvm", "knob": 1}, {"cann": "9"})
        context = RequestContext(
            manifest, "dvm", (torch.device("npu", 0),), {})

        uuid = self._uuid(context)

        self.assertEqual(
            uuid,
            ("torch_npu.dvm", manifest.schema,
             manifest.request_json, manifest.identity_json))

    def test_different_requests_yield_different_uuids(self):
        first = Manifest.create({"backend": "dvm", "knob": 1}, {"cann": "9"})
        second = Manifest.create({"backend": "dvm", "knob": 2}, {"cann": "9"})
        devices = (torch.device("npu", 0),)

        self.assertNotEqual(
            self._uuid(RequestContext(first, "dvm", devices, {})),
            self._uuid(RequestContext(second, "dvm", devices, {})))


if __name__ == "__main__":
    run_tests()
