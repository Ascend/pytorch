# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""NPU environment identity for native CompilePackage reuse.

A DVM compile package is reused whenever the native ``SystemInfo`` accepts the
environment, so these checks are the whole safety net for cross-chip and
cross-CANN reuse. On Ascend native collects no chip and no CANN and never takes
its device branch, which is why
``torch_npu.dynamo._package_compat.system_info`` exists and why
``CompilePackage.cache_entry`` relabels an owned package ``npu``.

The negative cases are pinned on purpose: without the ``npu`` label native skips
its device branch entirely, so a package from another chip is accepted.

Like every other adapter in this package, installation is process-global and
latching: once ``patches.install`` has run, a later call is a no-op. The cases
below therefore install once and never try to uninstall, exactly as production
does; only the ``SystemInfo`` class attribute, which the adapter mutates rather
than owns, is restored between tests.
"""
import dataclasses
import unittest

import torch
import torch._dynamo.config as dynamo_config
from torch._dynamo import package as native_package
from torch._dynamo.package import _BackendId
from torch._dynamo.precompile_context import PrecompileContext
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu._compat.version import CURRENT_VERSION
from torch_npu.dynamo._package_compat import system_info

RUN_NPU = torch.npu.is_available()
NPU = "npu"


def _variant(info, **changes):
    """``SystemInfo`` is frozen; build another host's environment from it."""
    return dataclasses.replace(info, **changes)


class _SystemInfoStateMixin:
    """Restore the one class attribute the adapter mutates in place."""

    def setUp(self):
        cls = native_package.SystemInfo
        self._gpus = tuple(cls.CHECK_GPUS)
        self._current = cls.__dict__["current"]

    def tearDown(self):
        cls = native_package.SystemInfo
        cls.CHECK_GPUS = self._gpus
        cls.current = self._current


class NativeBaselineTests(_SystemInfoStateMixin, TestCase):
    """What native records before the adapter fills the NPU gaps."""

    def test_native_check_gpus_covers_cuda_and_xpu_only(self):
        # The gap itself. If upstream ever adds "npu", this fails so that
        # install() is revisited instead of appending a duplicate.
        self.assertEqual(tuple(native_package.SystemInfo.CHECK_GPUS), ("cuda", "xpu"))

    def test_native_current_cannot_report_a_cann_version(self):
        # Native reads torch.version.npu, which does not exist, so adding "npu"
        # to the list alone still yields no toolkit version. This is the half of
        # the gap only an explicit collector closes, and the reason cache_entry
        # rewrites the field instead of trusting the default factory.
        native_package.SystemInfo.CHECK_GPUS = ("cuda", "xpu", NPU)
        self.assertIsNone(native_package.SystemInfo.current().toolkit_version)


class InstallTests(_SystemInfoStateMixin, TestCase):
    """``install`` extends the native check instead of replacing it."""

    def test_install_appends_npu_after_the_native_entries(self):
        system_info.install(native_package)
        self.assertEqual(tuple(native_package.SystemInfo.CHECK_GPUS), ("cuda", "xpu", NPU))

    def test_install_is_idempotent(self):
        # install() runs from wrapper init, lazy dynamo setup and package install;
        # a repeat must neither grow the tuple nor wrap current again.
        system_info.install(native_package)
        installed = native_package.SystemInfo.__dict__["current"]
        system_info.install(native_package)
        self.assertEqual(tuple(native_package.SystemInfo.CHECK_GPUS), ("cuda", "xpu", NPU))
        self.assertIs(native_package.SystemInfo.__dict__["current"], installed)

    def test_install_keeps_current_a_classmethod(self):
        # An entry's default SystemInfo comes from a dataclass default_factory
        # bound to SystemInfo.current, so it must stay callable through the class.
        system_info.install(native_package)
        self.assertIsInstance(native_package.SystemInfo.__dict__["current"], classmethod)
        self.assertIsInstance(native_package.SystemInfo.current(), native_package.SystemInfo)

    def test_install_repairs_a_restored_native_classmethod(self):
        # The marker lives on the shared function, not on the class, so putting
        # the native classmethod back must be repaired by the next install()
        # rather than silently leaving the gap open.
        system_info.install(native_package)
        installed = native_package.SystemInfo.__dict__["current"]
        native_package.SystemInfo.current = self._current
        system_info.install(native_package)
        self.assertIs(native_package.SystemInfo.__dict__["current"].__func__, installed.__func__)


class CompatibilityTests(_SystemInfoStateMixin, TestCase):
    """Native ``check_compatibility`` reached through an NPU environment."""

    def setUp(self):
        super().setUp()
        system_info.install(native_package)
        self.host = system_info.current(native_package.SystemInfo)

    def other_host(self):
        """Same process, a different Ascend product."""
        return _variant(self.host, gpu_name=(self.host.gpu_name or "Ascend910B3") + "x")

    @unittest.skipIf(not RUN_NPU, "requires npu")
    def test_collected_environment_carries_chip_and_cann(self):
        # Without both, every Ascend package matches every other one.
        self.assertIsNotNone(self.host.gpu_name)
        self.assertIsNotNone(self.host.toolkit_version)

    def test_same_environment_is_accepted(self):
        self.host.check_compatibility(system_info.current(native_package.SystemInfo), NPU)

    def test_different_chip_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "different GPU"):
            self.other_host().check_compatibility(self.host, NPU)

    def test_different_toolkit_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "different toolkit version"):
            _variant(self.host, toolkit_version="0.0.0-not-this-release").check_compatibility(self.host, NPU)

    def test_cpu_device_type_skips_the_device_branch(self):
        # Why cache_entry relabels the package: a package labelled "cpu" passes
        # even when the chip differs, because native only takes its device branch
        # for a type listed in CHECK_GPUS.
        self.other_host().check_compatibility(self.host, "cpu")

    def test_missing_accelerator_fails_the_device_branch_on_availability(self):
        # "cuda" is listed, so the branch runs and rejects on availability rather
        # than on the chip. Recorded so the label choice stays justified.
        with self.assertRaisesRegex(RuntimeError, "not available"):
            self.other_host().check_compatibility(self.host, "cuda")


class CollectorsTests(TestCase):
    """The two collectors degrade to ``None`` instead of raising."""

    def test_collectors_return_a_value_or_none(self):
        self.assertTrue(system_info.chip_name() is None or isinstance(system_info.chip_name(), str))
        self.assertTrue(system_info.cann_version() is None or isinstance(system_info.cann_version(), str))

    @unittest.skipIf(not RUN_NPU, "requires npu")
    def test_collectors_report_the_live_environment(self):
        from torch_npu.npu.utils import get_cann_version

        self.assertEqual(system_info.chip_name(), torch.npu.get_device_name())
        self.assertEqual(system_info.cann_version(), str(get_cann_version("CANN")))


class PackageLabelTests(_SystemInfoStateMixin, TestCase):
    """``cache_entry`` writes the environment and the device label.

    The package adapters only exist while ``caching_precompile`` is on, and their
    installation latches for the process, so this class turns the flag on and
    leaves the adapters in place.
    """

    @classmethod
    def setUpClass(cls):
        from torch_npu.dynamo._package_compat import patches

        cls._patches = patches
        cls._flag = dynamo_config.caching_precompile
        dynamo_config.caching_precompile = True
        system_info.install(native_package)
        cls._installed = patches.install()
        if not cls._installed or "cache_entry" not in (patches._package_originals or {}):
            raise unittest.SkipTest("package adapters unavailable on this PyTorch build")

    @classmethod
    def tearDownClass(cls):
        dynamo_config.caching_precompile = cls._flag

    def setUp(self):
        super().setUp()
        # Every case starts from a clean artifact registry, so ownership is
        # decided by exactly the records the case plants.
        PrecompileContext._backend_artifacts_by_key.clear()

    def tearDown(self):
        PrecompileContext._backend_artifacts_by_key.clear()
        super().tearDown()

    @staticmethod
    def _package(owned):
        def fn(x):
            return x + 1

        package = native_package.CompilePackage(fn)
        key = "backend_0"
        code_entry = next(iter(package._codes.values()))
        if key not in code_entry.backend_ids:
            code_entry.backend_ids.append(_BackendId(key))
        if owned:
            # Any DVM record marks the package as ours; has_dvm_records only
            # looks for the private field, it does not validate its contents.
            class _Artifact:
                def __init__(self):
                    self._npu_manifest = "marker"

            PrecompileContext._backend_artifacts_by_key[key] = _Artifact()
        return package

    def test_owned_package_is_labelled_npu(self):
        # Native derives the label from the first graph node and stops at the
        # first non-tensor value, so this correction is what lets the NPU
        # environment check run at all.
        self.assertEqual(self._package(True).cache_entry().device_type, NPU)

    @unittest.skipIf(not RUN_NPU, "requires npu")
    def test_owned_package_records_the_full_environment(self):
        entry = self._package(True).cache_entry()
        self.assertEqual(entry.system_info.gpu_name, system_info.chip_name())
        self.assertEqual(entry.system_info.toolkit_version, system_info.cann_version())

    def test_captured_default_factory_cannot_supply_the_cann_version(self):
        # The dataclass bound SystemInfo.current when the class was created, so a
        # fresh entry asks the *native* collector. It now reports the chip (the
        # device list is a class attribute) but still no CANN version, since
        # native reads torch.version.npu. That asymmetry is exactly why
        # cache_entry rewrites the field instead of trusting the factory.
        captured = native_package._DynamoCacheEntry.__dataclass_fields__["system_info"].default_factory
        self.assertIsNot(getattr(captured, "__func__", captured), system_info.current)
        self.assertIsNone(captured().toolkit_version)

    def test_unowned_package_keeps_its_native_device_label(self):
        # A package produced by any other backend must not be relabelled, or its
        # own reuse rules would change under a DVM adapter.
        self.assertNotEqual(self._package(False).cache_entry().device_type, NPU)

    @unittest.skipIf(not RUN_NPU, "requires npu")
    def test_unowned_npu_labelled_package_records_the_environment(self):
        # A package traced on NPU is labelled "npu" by the native graph, owned or
        # not, and npu is listed in CHECK_GPUS, so the consumer environment-checks
        # it like an owned one. The entry therefore has to record the chip and CANN
        # it was produced with, or the very host that saved it would reject it.
        package = self._package(False)
        # What tracing on NPU produces: the package records its device type.
        # COMPAT(< 2.15): upstream pytorch#196814 (first in torch 2.15)
        # replaced CompilePackage._device_type (str) with _device_types
        # (frozenset) folded through _collapse_device_types in cache_entry();
        # 2.13/2.14 keep the single native string.
        # CAN REMOVE the else branch when MIN_SUPPORTED_VERSION >= (2, 15)
        if CURRENT_VERSION >= (2, 15):
            package._device_types = frozenset({NPU})
        else:
            package._device_type = NPU
        entry = package.cache_entry()
        self.assertEqual(entry.device_type, NPU)  # filled, never relabelled
        self.assertEqual(entry.system_info.gpu_name, system_info.chip_name())
        self.assertEqual(entry.system_info.toolkit_version, system_info.cann_version())
        entry.check_versions()  # accepted by the environment that produced it


if __name__ == "__main__":
    run_tests()
