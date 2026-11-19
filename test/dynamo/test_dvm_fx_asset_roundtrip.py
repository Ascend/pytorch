# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# lintrunner: skip PYFMT
# Owner(s): ["module: dynamo"]
"""FX-asset archives must rebuild the graph they were exported from.

``export_fx``/``restore_fx`` (:mod:`torch_npu._inductor.dvm.fx_asset`)
carry the ``get_attr`` state a DVM FX fallback graph needs in another process.
The cross-process entry tests cover one small matmul graph end to end; what they
cannot show is whether the *structure* survives: module aliases, ``None``
children, parameters shared between modules, and storages shared across dtypes
-- the case ``_StatePickler`` exists for, since ``torch.save``'s typed-storage
path rejects it.

The disk-side contract lives here too: ``materialize`` must atomically replace
a missing or corrupt asset, and ``load_fx_asset`` must report damage instead of
returning a wrong module. These tests run against a private cache root, so they
never touch a real cache.
"""
import json
import os
import pickle
import tempfile
from pathlib import Path
from unittest import mock

import torch
import torch_npu  # noqa: F401  - registers torch.npu
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu._inductor.dvm.fx_asset import (
    FxAsset,
    export_fx,
    load_fx_asset,
    materialize,
    restore_fx,
)


def _inner_module(constant):
    """A GraphModule adding one archived constant to its input."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    value = graph.get_attr("constant")
    graph.output(graph.call_function(torch.add, (x, value)))
    root = torch.nn.Module()
    root.constant = constant
    return torch.fx.GraphModule(root, graph)


def _outer_module(children):
    """A GraphModule applying each child in order, aliasing included."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    for index, child in enumerate(children):
        call = graph.call_module(f"child{index}", (x if index == 0 else call,))
        x = call
    graph.output(x)
    root = torch.nn.Module()
    for index, child in enumerate(children):
        root.add_module(f"child{index}", child)
    return torch.fx.GraphModule(root, graph)


class _PrivateCacheRoot:
    """Point the inductor cache at a directory this test owns."""

    def __enter__(self):
        from torch._inductor.runtime.cache_dir_utils import cache_dir

        self._temporary = tempfile.TemporaryDirectory(prefix="npu-fx-roundtrip-")
        root = Path(self._temporary.name)
        self._patcher = mock.patch.dict(
            os.environ, {"TORCHINDUCTOR_CACHE_DIR": str(root)})
        self._patcher.start()
        # cache_dir() expands and memoizes the value in os.environ itself, so
        # the environment override has to be in place before anything asks.
        cache_dir()
        return root

    def __exit__(self, *args):
        self._patcher.stop()
        self._temporary.cleanup()


class TestFxAssetRoundTrip(TestCase):
    """Everything the archive format claims to keep, it really keeps."""

    def _roundtrip(self, gm):
        asset = export_fx(gm)
        materialize(asset)
        restored = load_fx_asset(asset.asset_id)
        return asset, asset.asset_id, restored

    def test_values_and_nested_graph_modules_survive(self):
        inner = _inner_module(torch.arange(4, dtype=torch.float32) * 2)
        outer = _outer_module([inner])
        x = torch.arange(4, dtype=torch.float32)

        _, _, restored = self._roundtrip(outer)

        # The inner GraphModule is rebuilt from its archived code, not from a
        # pickled module, so equality of output is the meaningful assertion.
        self.assertTrue(torch.equal(restored(x), outer(x)))

    def test_module_aliases_are_preserved_not_duplicated(self):
        inner = _inner_module(torch.ones(4))
        # One GraphModule object under two names: ``named_children()`` would
        # deduplicate it, ``_modules`` (which the exporter walks) keeps both.
        outer = _outer_module([inner, inner])
        x = torch.arange(4, dtype=torch.float32)

        _, _, restored = self._roundtrip(outer)

        self.assertTrue(torch.equal(restored(x), outer(x)))
        self.assertIs(restored.child0, restored.child1)

    def test_none_children_survive(self):
        inner = _inner_module(torch.ones(4))
        outer = _outer_module([inner])
        # A ``None`` child is recorded as a name only; it must come back as a
        # real ``None`` child, not disappear and shift its siblings. A plain
        # attribute assignment would bypass ``_modules``, so use add_module.
        outer.add_module("gap", None)

        _, _, restored = self._roundtrip(outer)

        self.assertIsNone(restored.gap)
        self.assertIsNotNone(getattr(restored, "child0", None))

    def test_parameter_aliasing_across_modules(self):
        shared = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
        first = _inner_module(shared)
        second = _inner_module(torch.ones(4))
        second.constant = shared  # same Parameter object, second owner

        _, _, restored = self._roundtrip(_outer_module([first, second]))

        # Serialization is one ``torch.save`` of all states together, which is
        # what lets a restored parameter be a single object again.
        self.assertIs(restored.child0.constant, restored.child1.constant)

    def test_shared_storage_across_dtypes(self):
        base = torch.arange(4, dtype=torch.float32)
        viewed = base.view(torch.int32)
        self.assertEqual(
            base.untyped_storage().data_ptr(), viewed.untyped_storage().data_ptr()
        )
        # Both tensors are named by get_attr nodes, which is what archives them
        # (the exporter only keeps attributes a node references).
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(graph.call_function(torch.add, (x, graph.get_attr("constant"))))
        graph.output(graph.call_function(torch.add, (x, graph.get_attr("viewed"))))
        graph.output(graph.call_function(torch.add, (x, graph.get_attr("constant"))))
        root = torch.nn.Module()
        root.constant = base
        root.viewed = viewed
        inner = torch.fx.GraphModule(root, graph)

        _, _, restored = self._roundtrip(inner)

        restored_base = restored.constant
        restored_view = restored.viewed
        self.assertEqual(restored_base.dtype, torch.float32)
        self.assertEqual(restored_view.dtype, torch.int32)
        # ``_StatePickler`` selects the v3 (untyped-storage) reducer precisely
        # so the two tensors keep sharing storage instead of being split into
        # incompatible typed copies.
        self.assertEqual(
            restored_base.untyped_storage().data_ptr(),
            restored_view.untyped_storage().data_ptr(),
        )

    def test_training_flag_and_non_persistent_buffers(self):
        inner = _inner_module(torch.ones(4))
        inner.register_buffer("npb", torch.zeros(4), persistent=False)
        inner.train(False)

        _, _, restored = self._roundtrip(inner)

        self.assertFalse(restored.training)
        self.assertIn("npb", restored._non_persistent_buffers_set)
        self.assertIn("npb", dict(restored.named_buffers()))

    def test_restore_rebuilds_a_fresh_module_each_call(self):
        inner = _inner_module(torch.ones(4))
        asset = export_fx(inner)
        materialize(asset)

        first = load_fx_asset(asset.asset_id)
        second = load_fx_asset(asset.asset_id)

        self.assertIsNot(first, second)


class TestUnportableGraphs(TestCase):
    """What the exporter refuses on structural grounds, not constant grounds."""

    def test_main_globals_are_refused(self):
        def imported_helper(x):
            return x + 1

        # The refusal keys on ``__module__ == "__main__"``, which is what a
        # function defined in a script (the common user case) reports.
        imported_helper.__module__ = "__main__"
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(graph.call_function(imported_helper, (x,)))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        from torch_npu._inductor.dvm.fx_asset import (
            UnsupportedFxConstant,
        )

        with self.assertRaises(UnsupportedFxConstant) as ctx:
            export_fx(gm)
        self.assertIn("__main__", str(ctx.exception))

    def test_triton_wrapper_imports_are_refused(self):
        from torch._higher_order_ops.triton_kernel_wrap import (
            triton_kernel_wrapper_functional,
        )

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(graph.call_function(triton_kernel_wrapper_functional, (x,)))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        from torch_npu._inductor.dvm.fx_asset import (
            UnsupportedFxConstant,
        )

        with self.assertRaises(UnsupportedFxConstant) as ctx:
            export_fx(gm)
        self.assertIn("NPU_TRITON_OUT_OF_SCOPE", str(ctx.exception))


class TestAssetValidation(TestCase):
    """``FxAsset.validate``/``restore_fx`` gate on the archived bytes."""

    def _asset_bytes(self, modules, states=None):
        metadata = json.dumps(
            {"schema": 1, "format": "torch_save", "modules": modules}
        ).encode()
        import io

        buffer = io.BytesIO()
        torch.save(states if states is not None else {"": {}}, buffer)
        return FxAsset(metadata, buffer.getvalue())

    def test_digest_mismatch_is_refused(self):
        asset = self._asset_bytes([{"path": "", "kind": "container", "type": "Module"}])
        wrong = "0" * 64
        from torch_npu.dynamo._package_compat.format import (
            PackageCompatibilityError,
        )

        with self.assertRaises(PackageCompatibilityError) as ctx:
            asset.validate(wrong)
        self.assertIn("NPU_FX_ASSET_DIGEST_MISMATCH", str(ctx.exception))

    def test_wrong_schema_is_refused(self):
        asset = self._asset_bytes(
            [{"path": "", "kind": "container", "type": "Module"}])
        metadata = json.loads(asset.metadata)
        metadata["schema"] = 999
        asset = FxAsset(json.dumps(metadata).encode(), asset.payload)

        from torch_npu.dynamo._package_compat.format import (
            PackageCompatibilityError,
        )

        with self.assertRaises(PackageCompatibilityError) as ctx:
            asset.validate()
        self.assertIn("NPU_FX_ASSET_SCHEMA_UNSUPPORTED", str(ctx.exception))

    def test_unsupported_module_kind_is_refused(self):
        asset = self._asset_bytes([{"path": "", "kind": "opaque"}])

        from torch_npu.dynamo._package_compat.format import (
            PackageCompatibilityError,
        )

        with self.assertRaises(PackageCompatibilityError) as ctx:
            asset.validate()
        self.assertIn("NPU_FX_MODULE_UNSUPPORTED", str(ctx.exception))

    def test_incomplete_state_is_refused_on_restore(self):
        # The metadata names two modules but the payload only carries state
        # for one: the restore must refuse rather than half-build a graph.
        from torch_npu.dynamo._package_compat.format import (
            PackageCompatibilityError,
        )

        inner = _inner_module(torch.ones(4))
        outer = _outer_module([inner, _inner_module(torch.zeros(4))])
        asset = export_fx(outer)
        metadata = json.loads(asset.metadata)
        kept = [spec for spec in metadata["modules"] if spec["path"] in ("", "child0")]
        trimmed_metadata = json.dumps(metadata | {"modules": kept}).encode()
        trimmed = FxAsset(trimmed_metadata, asset.payload)

        with self.assertRaises(PackageCompatibilityError) as ctx:
            restore_fx(trimmed)
        self.assertIn("NPU_FX_ASSET_INCOMPLETE", str(ctx.exception))

    def test_non_bytes_fields_are_refused(self):
        asset = FxAsset("not-bytes", b"payload")

        from torch_npu.dynamo._package_compat.format import (
            PackageCompatibilityError,
        )

        with self.assertRaises(PackageCompatibilityError) as ctx:
            asset.validate()
        self.assertIn("NPU_FX_ASSET_FORMAT_UNSUPPORTED", str(ctx.exception))


class TestMaterializeTolerance(TestCase):
    """A damaged disk must never turn a cache hit into a wrong answer."""

    def _exported(self):
        inner = _inner_module(torch.arange(4, dtype=torch.float32) * 3)
        asset = export_fx(inner)
        return asset, inner

    def test_missing_asset_file_is_rewritten(self):
        with _PrivateCacheRoot():
            asset, inner = self._exported()
            path = Path(materialize(asset))
            self.assertTrue(path.is_file())
            path.unlink()

            rewritten = Path(materialize(asset))

            self.assertTrue(rewritten.is_file())
            x = torch.arange(4, dtype=torch.float32)
            self.assertTrue(torch.equal(load_fx_asset(asset.asset_id)(x), inner(x)))

    def test_corrupt_asset_is_atomically_replaced(self):
        with _PrivateCacheRoot():
            asset, inner = self._exported()
            path = Path(materialize(asset))
            path.write_bytes(b"\x00garbage-not-an-asset")

            materialize(asset)

            x = torch.arange(4, dtype=torch.float32)
            self.assertTrue(torch.equal(load_fx_asset(asset.asset_id)(x), inner(x)))

    def _pickled_under(self, module):
        """An asset entry whose pickle names ``module`` instead of this one.

        This is what a cache written before the module moved looks like: the
        bytes are intact, but unpickling has to import a module that no longer
        exists. It must read as a miss, not as a compile failure.
        """
        asset, _ = self._exported()
        body = pickle.dumps(asset, protocol=4)
        stale = body.replace(
            b"torch_npu._inductor.dvm.fx_asset", module.encode(),
        )
        self.assertNotEqual(stale, body, "the pickle does not name this module")
        return asset, stale

    def test_a_pickle_from_a_moved_module_is_a_miss(self):
        with _PrivateCacheRoot():
            asset, stale = self._pickled_under("torch_npu.dynamo._package_compat.fx_fallback")
            path = Path(materialize(asset))
            path.write_bytes(stale)

            rewritten = Path(materialize(asset))

            self.assertTrue(rewritten.is_file())
            # The entry is usable again rather than poisoned by the old bytes.
            x = torch.arange(4, dtype=torch.float32)
            self.assertEqual(load_fx_asset(asset.asset_id)(x).shape, x.shape)

    def test_a_pickle_from_a_moved_module_is_reported_on_load(self):
        from torch_npu.dynamo._package_compat.format import (
            PackageCompatibilityError,
        )

        with _PrivateCacheRoot():
            asset, stale = self._pickled_under("torch_npu.dynamo._package_compat.fx_fallback")
            path = Path(materialize(asset))
            path.write_bytes(stale)

            with self.assertRaises(PackageCompatibilityError) as ctx:
                load_fx_asset(asset.asset_id)
            self.assertIn("NPU_FX_ASSET", str(ctx.exception))

    def test_a_resource_error_while_reading_is_not_a_miss(self):
        """An unreadable disk is a host problem, not a verdict on the bytes."""
        import errno

        with _PrivateCacheRoot():
            asset, _ = self._exported()

            with mock.patch.object(Path, "open",
                                   side_effect=OSError(errno.ENOSPC, "No space left")):
                with self.assertRaises(OSError):
                    materialize(asset)

    def test_load_fx_asset_reports_corruption(self):
        from torch_npu.dynamo._package_compat.format import (
            PackageCompatibilityError,
        )

        with _PrivateCacheRoot():
            asset, _ = self._exported()
            path = Path(materialize(asset))
            path.write_bytes(b"\x00garbage-not-an-asset")

            with self.assertRaises(PackageCompatibilityError) as ctx:
                load_fx_asset(asset.asset_id)
            self.assertIn("NPU_FX_ASSET_MISSING_OR_CORRUPT", str(ctx.exception))

    def test_live_id_expiry_is_reported(self):
        from torch_npu.dynamo._package_compat.format import (
            PackageCompatibilityError,
        )

        with self.assertRaises(PackageCompatibilityError) as ctx:
            load_fx_asset("live_never_registered")
        self.assertIn("NPU_LIVE_FX_ASSET_EXPIRED", str(ctx.exception))

    def test_invalid_asset_id_is_refused(self):
        from torch_npu.dynamo._package_compat.format import (
            PackageCompatibilityError,
        )

        with self.assertRaises(PackageCompatibilityError) as ctx:
            load_fx_asset("not-an-id")
        self.assertIn("NPU_FX_ASSET_ID_INVALID", str(ctx.exception))


if __name__ == "__main__":
    run_tests()
