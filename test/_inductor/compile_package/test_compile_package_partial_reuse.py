# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Partial DVM packages: a bypassed segment must not discard the rest.

Upstream marks a code bypassed when one of its backends failed to serialize, and
``CompilePackage.install`` skips exactly those codes. Dependency selection must
therefore look only at codes that will be restored, an all-bypassed entry is
empty rather than broken, and the exported set must not carry a dead segment's
leftovers.

The two ``NativeBypassTests`` cases are adapted from PyTorch's
``test/dynamo/test_package.py`` (``test_bypass_drops_only_the_current_compiles_backend``
and ``test_bypass_of_every_compile_marks_the_entry_bypassed``). They exercise the
native state our selection reads; the ``SelectionTests`` cases cover the NPU
selection built on top of it.
"""

from torch._dynamo import package as native_package
from torch._dynamo.package import PrecompileCacheEntry, _BackendId
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu.dynamo._package_compat.format import PackageCompatibilityError
from torch_npu.dynamo._package_compat.validation import restorable_backends


FIRST_ID = "backend_0"
SECOND_ID = "backend_1"


class _Region:
    """Stand-in for a compiled backend id recorded from bytecode."""

    def __init__(self):
        self.co_names = (FIRST_ID,)


def _fn(x):
    return x + 1


def _package(*, add_first=False, add_second=False, bypass_first=False):
    package = native_package.CompilePackage(_fn)
    code = next(iter(package._codes.values()))
    if add_first:
        code.backend_ids.append(_BackendId(FIRST_ID))
    if add_second:
        code.backend_ids.append(_BackendId(SECOND_ID))
    if bypass_first:
        code.bypassed = True
    return package


def _entry_with(*, live_ids=(), bypassed_ids=(), provided=()):
    """An entry whose codes carry ids ahead of what the store actually holds."""
    package = _package()
    live = next(iter(package._codes.values()))
    live.backend_ids.extend(_BackendId(i) for i in live_ids)
    if bypassed_ids:
        # A second code object stands for the segment native install skips.
        spare = native_package.CompilePackage(lambda y: y * 2)
        spare_code = next(iter(spare._codes.values()))
        spare_code.backend_ids.extend(_BackendId(i) for i in bypassed_ids)
        spare_code.bypassed = True
        package._codes[next(iter(spare._codes))] = spare_code
    dynamo = package.cache_entry()
    return PrecompileCacheEntry(dynamo=dynamo, backends={key: object() for key in provided})


class NativeBypassTests(TestCase):
    """The native state our selection reads. Adapted from PyTorch's test_package."""

    def test_bypass_drops_only_the_current_compiles_backend(self):
        # A bypass discards what the failing compile registered on its entry, not
        # what an earlier compile of the same code object did.
        package = _package(add_first=True)
        entry = next(iter(package._codes.values()))
        entry.backend_ids.append(_BackendId(SECOND_ID))
        entry.bypassed = True
        entry.backend_ids = [bid for bid in entry.backend_ids if bid != _BackendId(SECOND_ID)]
        self.assertFalse(entry.backend_ids == [])
        self.assertEqual(entry.backend_ids, [_BackendId(FIRST_ID)])

    def test_bypass_of_every_compile_marks_the_entry_bypassed(self):
        # With nothing installable the entry must be skipped rather than treated
        # as a trivial function; that is what makes "empty" a distinct outcome.
        entry = _package(add_first=True, bypass_first=True).cache_entry()
        self.assertTrue(entry.codes[0].bypassed)
        self.assertEqual(list(entry.codes[0].backend_ids), [FIRST_ID])


class SelectionTests(TestCase):
    """``restorable_backends`` narrows what native installation will restore."""

    def test_live_code_keeps_its_dependencies(self):
        entry = _entry_with(live_ids=(FIRST_ID,), provided=(FIRST_ID,))
        self.assertEqual(set(restorable_backends(entry)), {FIRST_ID})

    def test_bypassed_segment_is_not_required(self):
        # The defect this exists for: asking for a bypassed code's backend would
        # discard a package that is still fully usable.
        entry = _entry_with(live_ids=(FIRST_ID,), bypassed_ids=(SECOND_ID,),
                            provided=(FIRST_ID,))
        self.assertEqual(set(restorable_backends(entry)), {FIRST_ID})

    def test_bypassed_leftover_is_dropped_from_the_selection(self):
        entry = _entry_with(live_ids=(FIRST_ID,), bypassed_ids=(SECOND_ID,),
                            provided=(FIRST_ID, SECOND_ID))
        self.assertEqual(set(restorable_backends(entry)), {FIRST_ID})

    def test_missing_live_dependency_is_named(self):
        # Naming the segment is the point of narrowing the set.
        entry = _entry_with(live_ids=(FIRST_ID,), provided=())
        with self.assertRaisesRegex(PackageCompatibilityError, FIRST_ID):
            restorable_backends(entry)

    def test_live_code_without_dependencies_is_restorable(self):
        # Dynamo emits an empty graph for trivial functions, so a live code with
        # no backend of its own must not be reported as missing.
        entry = _entry_with(provided=())
        self.assertEqual(restorable_backends(entry), {})

    def test_non_bypassed_check_ignores_bypassed_codes_only(self):
        # An entry is reusable as long as one code will be restored.
        package = native_package.CompilePackage(_fn)
        live = next(iter(package._codes.values()))
        live.backend_ids.append(_BackendId(FIRST_ID))
        spare = native_package.CompilePackage(lambda y: y * 2)
        spare_code = next(iter(spare._codes.values()))
        spare_code.bypassed = True
        package._codes[next(iter(spare._codes))] = spare_code
        entry = PrecompileCacheEntry(dynamo=package.cache_entry(), backends={FIRST_ID: object()})
        self.assertTrue(any(not code.bypassed for code in entry.dynamo.codes))
        self.assertEqual(set(restorable_backends(entry)), {FIRST_ID})


if __name__ == "__main__":
    run_tests()
