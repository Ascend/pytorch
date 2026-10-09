# Copyright (c) 2026, Huawei Technologies Co., Ltd

"""Unit tests for the torch_npu._compat layer.

``CompatRegistryTest`` covers the registry itself and is pure logic, no NPU
device needed: only ``_impl.CURRENT_VERSION`` is patched to cover the version
branches. ``CompatModuleExportsTest`` checks the import surface the compat
modules declare in ``__all__``.
"""

import unittest

from torch_npu._compat import _impl
from torch_npu._compat._impl import CompatError, compat_impl, compat_impl_container


class CompatRegistryTest(unittest.TestCase):
    def setUp(self):
        compat_impl_container._reset_for_test()
        self._saved = _impl.CURRENT_VERSION
        _impl.CURRENT_VERSION = (2, 14)

    def tearDown(self):
        _impl.CURRENT_VERSION = self._saved
        compat_impl_container._reset_for_test()

    def set_version(self, version):
        _impl.CURRENT_VERSION = version

    # ---- matching and picking ----------------------------------------------

    def test_01_two_sides(self):
        @compat_impl(key="k", ge=(2, 15))
        def k_upstream():
            return "new"

        @compat_impl(key="k", lt=(2, 15))
        def k_local():
            return "old"

        self.set_version((2, 15))
        self.assertIs(compat_impl_container["k"].resolve(), k_upstream)
        self.set_version((2, 13))
        self.assertIs(compat_impl_container["k"].resolve(), k_local)

    def test_02_registration_order_wins(self):
        @compat_impl(key="k", judgement=True)
        def first():
            return 1

        self.assertIs(compat_impl_container["k"].resolve(), first)

    def test_03_version_sides_are_complementary(self):
        @compat_impl(key="k", ge=(2, 15))
        def k_upstream():
            return "new"

        @compat_impl(key="k", lt=(2, 15))
        def k_local():
            return "old"

        for version in ((2, 13), (2, 14), (2, 15), (2, 16), (3, 0)):
            self.set_version(version)
            self.assertIsNotNone(compat_impl_container["k"]._pick())

    def test_04_threshold_mismatch(self):
        @compat_impl(key="k", ge=(2, 15))
        def k_upstream():
            return "new"

        with self.assertRaises(CompatError):
            @compat_impl(key="k", lt=(2, 16))
            def k_local():
                return "old"

    def test_05_duplicate_direction(self):
        @compat_impl(key="k", ge=(2, 15))
        def k_upstream():
            return "new"

        with self.assertRaises(CompatError):
            @compat_impl(key="k", ge=(2, 15))
            def k_other():
                return "other"

    def test_06_more_than_two_sides(self):
        @compat_impl(key="k", ge=(2, 15))
        def k_upstream():
            return "new"

        @compat_impl(key="k", lt=(2, 15))
        def k_local():
            return "old"

        with self.assertRaises(CompatError):
            @compat_impl(key="k", lt=(2, 15))
            def k_extra():
                return "extra"

    def test_07_threshold_expired(self):
        with self.assertRaises(CompatError):
            @compat_impl(key="k", lt=(2, 13))
            def k_local():
                return "old"

    # ---- judgement (the escape hatch) --------------------------------------

    def test_08_judgement_hit(self):
        @compat_impl(key="k", judgement=True)
        def k_yes():
            return "yes"

        self.assertIs(compat_impl_container["k"].resolve(), k_yes)

    def test_09_judgement_overlap(self):
        @compat_impl(key="k", judgement=True)
        def k_yes():
            return "yes"

        with self.assertRaises(CompatError):
            @compat_impl(key="k", judgement=True)
            def k_also_yes():
                return "also"

    def test_10_mixing_forms(self):
        @compat_impl(key="k", ge=(2, 15))
        def k_upstream():
            return "new"

        with self.assertRaises(CompatError):
            @compat_impl(key="k", judgement=False)
            def k_other():
                return "other"

    # ---- resolve / apply ---------------------------------------------------

    def test_11_resolve_without_hit(self):
        @compat_impl(key="k", ge=(2, 15))
        def k_upstream():
            return "new"

        self.set_version((2, 13))
        with self.assertRaises(CompatError):
            compat_impl_container["k"].resolve()

    def test_12_apply_without_hit(self):
        @compat_impl(key="k", ge=(2, 15))
        def k_upstream():
            return "new"

        self.set_version((2, 13))
        self.assertIsNone(compat_impl_container["k"].apply())

    def test_13_apply_calls_the_function(self):
        calls = []

        @compat_impl(key="k", lt=(2, 15))
        def k_local():
            calls.append(1)
            return "bound"

        self.assertEqual(compat_impl_container["k"].apply(), "bound")
        self.assertEqual(calls, [1])

    def test_14_two_class_thunk_is_returned(self):
        @compat_impl(key="k", ge=(2, 15))
        def k_upstream(a, b):
            return a + b

        self.set_version((2, 15))
        fn = compat_impl_container["k"].resolve()
        self.assertEqual(fn(1, 2), 3)

    def test_15_one_class_thunk_is_called(self):
        @compat_impl(key="k", lt=(2, 15))
        def k_local():
            def bound():
                return "bound"

            return bound

        # One-class / mixed: apply() returns the thing to bind, not its result.
        bound_fn = compat_impl_container["k"].apply()
        self.assertTrue(callable(bound_fn))
        self.assertEqual(bound_fn(), "bound")

    # ---- validation and container ------------------------------------------

    def test_16_same_key_different_module(self):
        compat_impl_container.point("k", "module.a")
        with self.assertRaises(CompatError):
            compat_impl_container.point("k", "module.b")._add(
                _impl._Side("k", "module.b", lambda: None, _impl.LT, (2, 15), None)
            )

    def test_17_threshold_type(self):
        for bad in ((2, "15"), (2, 15, 0), 2.15, [2, 15]):
            with self.assertRaises(CompatError):
                compat_impl(key="k", lt=bad)

    def test_18_bad_key_or_condition_count(self):
        with self.assertRaises(CompatError):
            compat_impl(key="", lt=(2, 15))
        with self.assertRaises(CompatError):
            compat_impl(key=None, lt=(2, 15))
        with self.assertRaises(CompatError):
            compat_impl(key="k")
        with self.assertRaises(CompatError):
            compat_impl(key="k", ge=(2, 15), lt=(2, 15))

    def test_19_decorator_returns_function(self):
        @compat_impl(key="k", lt=(2, 15))
        def k_local():
            return "old"

        self.assertTrue(callable(k_local))
        self.assertEqual(k_local(), "old")

    def test_20_unknown_key(self):
        with self.assertRaises(CompatError):
            compat_impl_container["not-registered"]

    def test_21_side_condition_is_required(self):
        # A side needs exactly one of ge / lt / judgement. Giving none (a bare
        # key, or lt=None on its own) must fail loudly here rather than build a
        # versioned side whose version is None and blow up later at pick time.
        for kwargs in ({}, {"ge": None}, {"lt": None}, {"ge": None, "lt": None},
                       {"judgement": None}):
            with self.assertRaises(CompatError):
                compat_impl(key="k", **kwargs)


class CompatModuleExportsTest(unittest.TestCase):
    """Every name a compat module declares in ``__all__`` must exist.

    A name declared but missing means the binding line at the end of a compat
    module was dropped while its ``__all__`` entry stayed behind; nothing else
    in the test suite would notice.
    """

    def test_declared_names_exist(self):
        from torch_npu._compat import (
            _impl,
            accelerator,
            distributed,
            dynamo,
            inductor,
            version,
        )

        for module in (_impl, version, accelerator, dynamo, distributed, inductor):
            self.assertTrue(
                hasattr(module, "__all__"),
                f"{module.__name__} does not declare __all__",
            )
            for name in module.__all__:
                self.assertTrue(
                    hasattr(module, name),
                    f"{module.__name__}.{name} is declared in __all__ but missing",
                )


if __name__ == "__main__":
    unittest.main()
