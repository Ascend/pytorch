# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# lintrunner: skip PYFMT
# Owner(s): ["module: dynamo"]
"""DVM FX-archive persistence gate: a live asset must never reach a cache.

A DVM FX fallback graph owns an archive of its generated code and real
constants (:mod:`torch_npu._inductor.dvm.fx_asset`). When the graph
cannot be archived -- an unsupported constant, a non-base NPU storage format, or
resource exhaustion -- ``register_fx_asset`` degrades to a ``LiveFxAsset`` that
exists only in this process, and ``patches.graph_init`` marks the resulting
``CompiledFxGraph`` uncacheable so that no consumer can ever try to restore a
``live_<uuid>`` id.

These tests assert that contract directly instead of through a full DVM compile,
because the property under test is *what gets written*, not what gets computed.
They need no NPU device.

Not decorated with ``@runIfVersion``: the behaviour is version-independent, and a
decorated file is dropped entirely by ``ci/access_control/version_filter.py``
when the job runs without ``--between_version``.
"""
import copy
import json
import pickle

import torch
import torch_npu  # noqa: F401
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_npu._inductor.dvm.fx_asset import (
    FxAsset,
    LiveFxAsset,
    UnresolvedFxConstant,
    UnsupportedFxConstant,
    _check_value,
    register_fx_asset,
)
from torch_npu.dynamo._package_compat.format import PackageCompatibilityError


class _TensorSubclass(torch.Tensor):
    """``_check_value`` refuses a Tensor subclass as an FX constant."""


class _Opaque:
    """An attribute value the archive format cannot reproduce."""


class _NpuRecord:
    """Minimal stand-in for a graph record that ``walk_objects`` can traverse.

    Every field is an instance attribute: ``walk_objects`` descends into
    ``vars(item)``, so a class-level attribute is invisible to it and a record
    built that way would pass validation for the wrong reason.
    """

    device_types = frozenset({"npu"})
    source_code = ""


class _RecordWithMarker:
    """A cache record holding the uncacheable marker, as cache writes see it.

    Module-level on purpose: ``pickle`` refuses a locally defined class with an
    ``AttributeError`` before it ever reaches ``__reduce__``, so a class defined
    inside the test body would not exercise the marker at all.
    """

    def __init__(self):
        from torch_npu._inductor.dvm.fx_asset import Uncacheable

        self._npu_uncacheable = Uncacheable()
        self.wrapper = "def call(): ..."


class _StubGraph:
    """Stand-in for the ``GraphLowering`` that owns an archive."""

    def __init__(self):
        self._npu_fx_assets = {}


def _graph_module_with_unarchivable_constant():
    """A GraphModule holding one ``get_attr`` value the exporter must refuse.

    ``_check_value`` walks both ``_parameters``/``_buffers`` and the attributes
    named by ``get_attr`` nodes. An opaque object in a buffer reaches the final
    refusal branch before any NPU-specific check, so this exercises the
    degradation path on a CPU-only host.
    """
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    holder = graph.get_attr("constant")
    graph.output(graph.call_function(torch.add, (x, holder)))

    # ``GraphModule.__init__`` resolves every ``get_attr`` target on the root
    # module while it builds, so the attribute has to exist *before* the module
    # is constructed; assigning it afterwards raises AttributeError there.
    root = torch.nn.Module()
    root.constant = _Opaque()
    return torch.fx.GraphModule(root, graph)


class TestConstantGate(TestCase):
    """The archive format rejects exactly what it cannot reproduce."""

    def test_tensor_subclass_is_refused(self):
        with self.assertRaises(UnsupportedFxConstant):
            _check_value(_TensorSubclass(torch.randn(2)), "test.value")

    def test_opaque_attribute_is_refused(self):
        with self.assertRaises(UnsupportedFxConstant):
            _check_value(_Opaque(), "test.value")

    def test_plain_tensor_is_accepted(self):
        _check_value(torch.randn(2), "test.value")

    def test_unresolved_constant_is_a_separate_refusal(self):
        """A meta tensor is *not* an ``UnsupportedFxConstant``.

        ``UnresolvedFxConstant`` is a sibling, not a subclass: no live fallback
        can invent a real value, so the two must not be collapsed. Pinned here
        because that distinction is easy to lose in a refactor, and because the
        review of PR #49420 proposed merging them.
        """
        with self.assertRaisesRegex(
            UnresolvedFxConstant, r"^NPU_FX_CONSTANT_HAS_NO_REAL_VALUE: test.value"
        ):
            _check_value(torch.empty(2, device="meta"), "test.value")
        self.assertFalse(issubclass(UnresolvedFxConstant, UnsupportedFxConstant))

    def test_non_dict_metadata_is_a_format_error(self):
        # json.loads returns any JSON value; a dict is the only shape export
        # writes, and validate() must name the shape instead of crashing.
        asset = FxAsset(metadata=json.dumps([1, 2]).encode(), payload=b"")
        with self.assertRaisesRegex(
            PackageCompatibilityError, "NPU_FX_ASSET_FORMAT_UNSUPPORTED"
        ):
            asset.validate()


class TestLiveAssetDegradation(TestCase):
    """An unarchivable graph stays correct but is never persisted."""

    def test_refusal_degrades_instead_of_raising(self):
        gm = _graph_module_with_unarchivable_constant()
        graph = _StubGraph()

        asset_id = register_fx_asset(gm, graph)

        self.assertTrue(asset_id.startswith("live_"), asset_id)
        self.assertIsInstance(graph._npu_fx_assets[asset_id], LiveFxAsset)

    def test_live_ids_are_not_stable(self):
        """Two exports of one graph disagree, so dedup can never hit.

        This is the behaviour flagged on PR #49420: the id feeds ``kernel_key``
        in ``NpuDvmScheduling.define_kernel``, so a live asset misses
        ``src_to_kernel`` and the same kernel is emitted again under a new name.
        """
        gm = _graph_module_with_unarchivable_constant()

        first = register_fx_asset(gm, _StubGraph())
        second = register_fx_asset(gm, _StubGraph())

        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("live_"))
        self.assertTrue(second.startswith("live_"))

    def test_live_asset_records_its_reason(self):
        gm = _graph_module_with_unarchivable_constant()
        graph = _StubGraph()

        asset_id = register_fx_asset(gm, graph)

        self.assertTrue(getattr(graph._npu_fx_assets[asset_id], "reason", None))


class TestLiveAssetIsNotPortable(TestCase):
    """The archive itself refuses to travel, which is what makes it safe."""

    def test_pickle_refused(self):
        asset = LiveFxAsset(torch.nn.Module())
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(asset)

    def test_deepcopy_preserved(self):
        """Native cache saves deepcopy the record before serializing it."""
        asset = LiveFxAsset(torch.nn.Module())
        self.assertIs(copy.deepcopy(asset), asset)


class TestUncacheableMarker(TestCase):
    """``_npu_uncacheable`` is the mechanism that makes a write a no-op.

    ``patches.graph_init`` attaches it to a ``CompiledFxGraph`` whose assets
    cannot be archived. Every FX and AOTAutograd write path pickles the record,
    so a ``PicklingError`` from ``__reduce__`` is what stops the write.
    """

    def test_marker_rejects_pickle(self):
        from torch_npu._inductor.dvm.fx_asset import Uncacheable

        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(Uncacheable())

    def test_marker_survives_deepcopy(self):
        from torch_npu._inductor.dvm.fx_asset import Uncacheable

        marker = Uncacheable()
        self.assertIs(copy.deepcopy(marker), marker)

    def test_marker_poisons_its_record(self):
        """One marker anywhere in the record stops the whole pickle."""
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(_RecordWithMarker())


class TestValidatorRejectsLiveAsset(TestCase):
    """``validate_dvm_assets`` is what ``patches._graph_init`` relies on.

    It runs on the write path, before any cache entry exists, so a consumer
    never observes a ``live_`` id at all.
    """

    def test_live_asset_is_not_portable(self):
        from torch_npu.dynamo._package_compat.format import (
            PackageCompatibilityError,
        )
        from torch_npu._inductor.dvm.persistence import validate_dvm_assets

        record = _NpuRecord()
        record._npu_fx_assets = {"live_deadbeef": LiveFxAsset(torch.nn.Module())}

        with self.assertRaises(PackageCompatibilityError) as ctx:
            validate_dvm_assets(record)
        self.assertIn("NPU_FX_ASSET_NOT_PORTABLE", str(ctx.exception))

    def test_uncacheable_record_is_not_portable(self):
        from torch_npu.dynamo._package_compat.format import (
            PackageCompatibilityError,
        )
        from torch_npu._inductor.dvm.persistence import validate_dvm_assets

        record = _NpuRecord()
        record._npu_persistable = False

        with self.assertRaises(PackageCompatibilityError) as ctx:
            validate_dvm_assets(record)
        self.assertIn("NPU_ARTIFACT_NOT_PORTABLE", str(ctx.exception))


class TestForceDisableCachesOnIdentityLoss(TestCase):
    """An unkeyable request must not touch either cache."""

    def test_uncacheable_request_disables_caches(self):
        from torch._inductor import config

        from torch_npu._inductor.dvm.persistence import (
            CompileRequest,
            RequestContext,
            request_scope,
        )

        request = CompileRequest({"npu_backend": "dvm"})
        context = RequestContext.uncacheable(
            "dvm",
            (torch.device("npu", 0),),
            "NPU_REQUEST_UNCACHEABLE",
            request=request,
        )
        self.assertFalse(context.persistable)
        self.assertFalse(config.force_disable_caches)

        with request_scope(context):
            self.assertTrue(config.force_disable_caches)
        # Scoped, not permanent...
        self.assertFalse(config.force_disable_caches)
        # ...and also recorded for a deferred backward replayed later.
        self.assertTrue(request.options.get("force_disable_caches"))


class TestCacheWriterTolerance(TestCase):
    """The write paths this marker relies on must stay tolerant.

    ``Uncacheable`` works by raising ``PicklingError`` from ``__reduce__``.
    That is only a no-op while the writers swallow the exception, so the
    assumption is asserted here rather than left implicit: if upstream ever
    stops tolerating it, the marker becomes a compile failure, not a bypass.
    """

    def test_fx_graph_cache_swallows_pickle_failure(self):
        import inspect

        from torch._inductor.codecache import FxGraphCache

        source = inspect.getsource(FxGraphCache._save_graph)
        self.assertIn("pickle.dumps", source)
        self.assertIn("except Exception", source)

    def test_aot_autograd_save_error_keys_on_strict_autograd_cache(self):
        """``_handle_save_error`` raises only for ``strict_autograd_cache``.

        Worth pinning: ``strict_precompile`` alone does *not* make the
        AOTAutograd save path raise, so the marker stays a silent bypass there
        unless ``strict_autograd_cache`` is set as well.
        """
        import inspect

        from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache

        source = inspect.getsource(AOTAutogradCache._handle_save_error)
        self.assertIn("strict_autograd_cache", source)
        self.assertNotIn("strict_precompile", source)

    def test_marker_bridges_the_gap_strict_precompile_leaves(self):
        """With ``strict_autograd_cache`` on, a marker really does fail the save.

        Asserts the *mechanism*, not a compile: the exception the writer would
        re-raise is the one ``Uncacheable`` produces.
        """
        from torch._functorch import config as functorch_config

        with functorch_config.patch(strict_autograd_cache=True):
            self.assertTrue(functorch_config.strict_autograd_cache)
            with self.assertRaises(pickle.PicklingError):
                pickle.dumps(_RecordWithMarker())


class TestTritonBoundary(TestCase):
    """Triton dependencies must stay out of a DVM FX archive.

    Native generated wrappers may carry Triton, but this archive does not
    restore its runtime side tables, so an asset that secretly depends on a
    Triton kernel must be refused at export time and at validation time.
    """

    def _refused(self, source):
        from torch_npu._inductor.dvm.fx_asset import (
            check_no_triton_source,
        )
        from torch_npu.dynamo._package_compat.format import (
            PackageCompatibilityError,
        )

        with self.assertRaises(PackageCompatibilityError) as ctx:
            check_no_triton_source(source)
        self.assertIn("NPU_TRITON_OUT_OF_SCOPE", str(ctx.exception))

    def test_import_triton_is_refused(self):
        self._refused("import triton\n")

    def test_import_triton_submodule_is_refused(self):
        self._refused("import triton.language as tl\n")

    def test_import_from_the_wrapper_module_is_refused(self):
        self._refused(
            "from torch._higher_order_ops.triton_kernel_wrap import wrap_triton\n")

    def test_attribute_access_is_refused(self):
        self._refused("def f(x):\n    return x.triton_kernel_wrapper_functional\n")

    def test_bare_wrapper_name_is_refused(self):
        self._refused("def f():\n    return triton_kernel_wrapper_mutation\n")

    def test_clean_source_passes(self):
        from torch_npu._inductor.dvm.fx_asset import (
            check_no_triton_source,
        )

        # Comments and string literals never reach the AST walk, so a mention
        # of triton there is not a dependency.
        check_no_triton_source(
            "# a comment about triton\n"
            "NAME = 'triton'\n"
            "def f(x):\n    return x + 1\n")

    def test_a_static_autotuner_bundle_is_refused(self):
        """``check_dvm_graph`` rejects static Triton autotuners on a DVM graph."""
        from types import SimpleNamespace

        from torch_npu.dynamo._package_compat.format import (
            Manifest,
            PackageCompatibilityError,
        )
        from torch_npu._inductor.dvm.persistence import check_dvm_graph

        record = _NpuRecord()
        record._npu_manifest = Manifest.create({"backend": "dvm"}, {})
        record._triton_bundle = SimpleNamespace(static_autotuners=[object()])

        with self.assertRaises(PackageCompatibilityError) as ctx:
            check_dvm_graph(record)
        self.assertIn("NPU_STATIC_TRITON_OUT_OF_SCOPE", str(ctx.exception))

    def test_a_graph_without_a_manifest_is_refused(self):
        from torch_npu.dynamo._package_compat.format import (
            PackageCompatibilityError,
        )
        from torch_npu._inductor.dvm.persistence import check_dvm_graph

        with self.assertRaises(PackageCompatibilityError) as ctx:
            check_dvm_graph(_NpuRecord())
        self.assertIn("NPU_MANIFEST_MISSING", str(ctx.exception))


class TestResourceErrorDegradation(TestCase):
    """Resource exhaustion degrades to a live asset; programming errors raise.

    ``register_fx_asset`` treats ``MemoryError``/ENOSPC-class failures like an
    unsupported constant: the graph stays correct in this process as a
    ``LiveFxAsset`` and the bypass is counted. Any other ``RuntimeError`` is a
    bug in the exporter itself and must not be silenced.
    """

    def _counter(self):
        from torch._dynamo.utils import counters

        return counters["npu_package"].get("NPU_FX_PERSISTENCE_BYPASS", 0)

    def test_memory_error_degrades_to_a_live_asset(self):
        from unittest import mock

        gm = _graph_module_with_unarchivable_constant()
        graph = _StubGraph()
        before = self._counter()

        with mock.patch(
            "torch_npu._inductor.dvm.fx_asset.export_fx",
            side_effect=MemoryError(),
        ):
            asset_id = register_fx_asset(gm, graph)

        self.assertTrue(asset_id.startswith("live_"))
        self.assertIsInstance(graph._npu_fx_assets[asset_id], LiveFxAsset)
        self.assertEqual(self._counter(), before + 1)

    def test_disk_exhaustion_degrades_to_a_live_asset(self):
        import errno
        from unittest import mock

        gm = _graph_module_with_unarchivable_constant()
        graph = _StubGraph()
        before = self._counter()

        with mock.patch(
            "torch_npu._inductor.dvm.fx_asset.materialize",
            side_effect=OSError(errno.ENOSPC, "No space left on device"),
        ):
            asset_id = register_fx_asset(gm, graph)

        self.assertTrue(asset_id.startswith("live_"))
        self.assertIsInstance(graph._npu_fx_assets[asset_id], LiveFxAsset)
        self.assertEqual(self._counter(), before + 1)

    def test_a_programming_error_is_raised(self):
        from unittest import mock

        gm = _graph_module_with_unarchivable_constant()
        graph = _StubGraph()

        with mock.patch(
            "torch_npu._inductor.dvm.fx_asset.export_fx",
            side_effect=RuntimeError("exporter bug"),
        ):
            with self.assertRaises(RuntimeError):
                register_fx_asset(gm, graph)
        self.assertEqual(graph._npu_fx_assets, {})

    def _run_dvm_debug_dump(self, dump_error):
        """Run the DVM FX debug dump against a throwaway cache root.

        ``NpuDvmScheduling._dump_fx_graph_for_fallback`` is self-contained in
        the DVM backend: it serializes the graph through ``to_folder`` instead
        of delegating to the MLIR backend's ``NpuMetaScheduling``, so the dump
        step is driven by patching ``to_folder`` where the DVM module binds it.
        """
        import os
        import tempfile
        from unittest import mock

        from torch_npu._inductor.dvm import mlir_fusion

        with tempfile.TemporaryDirectory() as cache_root, mock.patch.dict(
            os.environ, {"TORCHINDUCTOR_CACHE_DIR": cache_root}
        ), mock.patch.object(mlir_fusion, "to_folder", side_effect=dump_error):
            mlir_fusion.NpuDvmScheduling._dump_fx_graph_for_fallback(
                None, mock.Mock(), mock.Mock(index=0), "deadbeef")

    def test_the_debug_dump_tolerates_resource_errors(self):
        """A failed FX debug dump must not undo a persistence bypass.

        ``NpuDvmScheduling._dump_fx_graph_for_fallback`` keeps its debug
        output path but swallows exactly the recoverable resource errors, so an
        ENOSPC while dumping cannot turn a successful live fallback into a
        crash.
        """
        import errno

        self._run_dvm_debug_dump(
            OSError(errno.ENOSPC, "No space left on device"))

    def test_the_debug_dump_raises_programming_errors(self):
        with self.assertRaises(RuntimeError):
            self._run_dvm_debug_dump(RuntimeError("dump bug"))


if __name__ == "__main__":
    run_tests()
