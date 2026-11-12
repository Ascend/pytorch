import copy
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("TORCHINDUCTOR_NPU_BACKEND", "triton_experimental")

import torch
import torch_npu  # noqa: F401
import torch_npu._inductor  # noqa: F401
from torch._inductor import config
from torch._inductor.codecache import compiled_fx_graph_hash
from torch._inductor.codegen.common import custom_backend_codegen_configs
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.runtime.runtime_utils import triton_hash_to_path_key
from torch._inductor.triton_bundler import TritonBundler
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._ordered_set import OrderedSet

from torch_npu._inductor.triton_experimental import config as ncfg
from torch_npu._inductor.triton_experimental import fx_passes
from torch_npu._inductor.triton_experimental import npu_triton_heuristics as heuristics
from torch_npu._inductor.triton_experimental.codegen.triton import NPUTritonCSEVariable, NPUTritonKernel
from torch_npu._inductor.triton_experimental.npu_triton_heuristics import NPUCachingAutotuner


class TestPackageIdentity(TestCase):
    def test_reduction_config_matches_key_codegen_and_metadata(self):
        # Namespaced compile options own the "npu" slot on newer PyTorch;
        # the backend config remains registered under its module name.
        self.assertIn(ncfg, custom_backend_codegen_configs.values())
        # Modules are already imported: a runtime override must not retain an
        # import-time default in codegen or in the saved kernel configuration.
        gm = torch.fx.symbolic_trace(lambda x: x.sum())
        inputs = [torch.empty(8)]
        kernel = object.__new__(NPUTritonKernel)
        kernel.inside_reduction = True
        kernel.persistent_reduction = False
        value = object.__new__(NPUTritonCSEVariable)
        value._npu_pad = ("c", 0.0)
        keys = []
        for enabled in (False, True):
            with ncfg.patch(elide_reduction_where=enabled):
                keys.append(compiled_fx_graph_hash(gm, inputs, {}, [])[0])
                metadata = NPUTritonKernel.inductor_meta_common()
                self.assertEqual(metadata["npu_backend_config"]["elide_reduction_where"], enabled)
                with mock.patch.object(
                    TritonKernel,
                    "reduction",
                    side_effect=lambda *args: getattr(kernel, "_npu_elide_reduction_masks", False),
                ):
                    self.assertEqual(kernel.reduction(torch.float32, torch.float32, "sum", value), enabled)
            with ncfg.patch(elide_reduction_where=not enabled):
                self.assertEqual(heuristics._runtime_config(metadata, "elide_reduction_where"), enabled)
        self.assertNotEqual(keys[0], keys[1])

    def test_template_kernels_carry_backend_config(self):
        from torch_npu._inductor.triton_experimental.flex_attention.flex_attention_template import (
            ExperimentalTemplateKernel,
        )

        with ncfg.patch(enable_vf_fusion=True, vf_merge_level=1):
            metadata = ExperimentalTemplateKernel.inductor_meta_common()
        self.assertTrue(metadata["npu_backend_config"]["enable_vf_fusion"])
        self.assertEqual(metadata["npu_backend_config"]["vf_merge_level"], 1)

    def test_tile_alignment_uses_producer_configuration(self):
        with ncfg.patch(tile_align=32):
            metadata = NPUTritonKernel.inductor_meta_common()
        fn = SimpleNamespace(fn=lambda x, XBLOCK: None)
        candidates = []
        for current in (32, 64):
            with (
                ncfg.patch(tile_align=current),
                mock.patch("inspect.getsource", return_value="assert x.shape == (XBLOCK,)"),
                mock.patch.object(heuristics, "NPUCachingAutotuner", return_value=SimpleNamespace()) as tuner,
            ):
                configs = [heuristics.Config({"XBLOCK": block}) for block in (256, 512)]
                heuristics.cached_autotune(
                    {"x": 65}, configs, {}, heuristics.HeuristicType.POINTWISE,
                    filename="/outside/cache/kernel.py", inductor_meta=metadata.copy(),
                )(fn)
                candidates.append(tuner.call_args.kwargs["configs"][0].kwargs["XBLOCK"])
        self.assertEqual(candidates, [96, 96])

    def test_legacy_metadata_uses_current_tile_alignment(self):
        fn = SimpleNamespace(fn=lambda x, XBLOCK: None)
        with (
            ncfg.patch(tile_align=64),
            mock.patch("inspect.getsource", return_value="assert x.shape == (XBLOCK,)"),
        ):
            configs = [heuristics.Config({"XBLOCK": 256})]
            result = heuristics._add_tile_consistent_config(
                {"x": 65}, configs, fn, heuristics.HeuristicType.POINTWISE, {}
            )
        self.assertEqual(result[0].kwargs["XBLOCK"], 128)


class TestPackageSystemIdentity(TestCase):
    def test_package_versions_change_native_fx_and_aot_keys(self):
        from torch._functorch._aot_autograd.autograd_cache import autograd_cache_key
        from torch._functorch._aot_autograd.schemas import AOTConfig
        from torch._guards import TracingContext, tracing
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        aot_config = AOTConfig(
            fw_compiler=None, bw_compiler=None, partition_fn=None, decompositions={},
            num_params_buffers=0, aot_id=0, keep_inference_input_mutations=False,
        )
        gm = torch.fx.symbolic_trace(lambda x: x + 1)
        inputs = [torch.empty(2)]
        keys = []
        with (
            config.patch(post_grad_custom_post_pass=[fx_passes.TeSourcePinPass()]),
            tracing(TracingContext(FakeTensorMode(shape_env=ShapeEnv()))),
        ):
            for pta, commit, ta in (
                ("2.15.0", "build-a", "3.6.0"),
                ("2.15.0", "build-a", "3.6.0"),
                ("2.15.1", "build-a", "3.6.0"),
                ("2.15.0", "build-b", "3.6.0"),
                ("2.15.0", "build-a", "3.6.0.post2"),
                ("2.15.0+gitbuild-b", "build-b", "3.6.0"),
            ):
                with (
                    mock.patch.object(torch_npu, "__version__", pta),
                    mock.patch("torch_npu.version.git_version", commit),
                    mock.patch("importlib.metadata.version", return_value=ta),
                ):
                    keys.append((
                        compiled_fx_graph_hash(gm, inputs, {}, [])[0],
                        autograd_cache_key(copy.deepcopy(gm), inputs, aot_config)[0],
                    ))
        self.assertEqual(keys[0], keys[1])
        # A separate commit field does not invalidate an unchanged package version.
        self.assertEqual(keys[0], keys[3])
        for layer in (0, 1):
            self.assertEqual(len({key[layer] for key in keys}), 4)

    def test_source_install_without_distribution_metadata(self):
        from importlib.metadata import PackageNotFoundError

        with mock.patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            identity = fx_passes.TeSourcePinPass().uuid()
            self.assertEqual(identity, fx_passes.TeSourcePinPass().uuid())
        self.assertIsInstance(identity, bytes)


class TestPackageKernelRegistration(TestCase):
    def autotuner(self):
        tuner = object.__new__(NPUCachingAutotuner)
        tuner.compile_results = []
        tuner.launchers = []
        tuner.configs = [object()]
        tuner.triton_meta = {"device": 2}
        tuner.inductor_meta = {"kernel_name": "package_test"}
        return tuner

    def test_direct_candidate_compilation_carries_config_and_registers_kernel(self):
        tuner = self.autotuner()
        tuner.heuristic_type = heuristics.HeuristicType.POINTWISE
        tuner.fn = SimpleNamespace(arg_names=["x", "XBLOCK"], constexprs=[1])
        tuner.triton_meta.update(constants={}, signature={"x": "*fp32"}, configs=[object()])
        tuner.device_props = SimpleNamespace(type="npu", cc="Ascend910B3", warp_size=32)
        with ncfg.patch(vf_merge_level=1, enable_vf_fusion=True):
            tuner.inductor_meta.update(NPUTritonKernel.inductor_meta_common())
        binary = SimpleNamespace(hash="a" * 64)
        with (
            ncfg.patch(vf_merge_level=None, enable_vf_fusion=False),
            mock.patch.object(TritonBundler, "_entries", []),
            mock.patch.object(heuristics, "ASTSource"),
            mock.patch.object(heuristics, "NPUTritonCompileResult"),
            mock.patch.object(heuristics.NPUStaticTritonCompileResult,
                              "can_statically_launch", return_value=None),
            mock.patch("torch._inductor.runtime.triton_helpers.set_driver_to_gpu"),
            mock.patch.object(heuristics.triton, "compile", return_value=binary) as compile_,
        ):
            tuner._precompile_config(heuristics.Config({"XBLOCK": 256}))
            self.assertEqual(len(TritonBundler._entries), 1)
            self.assertEqual(TritonBundler._entries[0].kernel_hash,
                             triton_hash_to_path_key(binary.hash))
            self.assertTrue(compile_.call_args.kwargs["options"]["enable_vf_fusion"])
            self.assertEqual(compile_.call_args.kwargs["options"]["vf_merge_level"], 1)

    def test_mixed_kernel_winners_are_bundled(self):
        tuner = self.autotuner()
        launcher = mock.Mock(return_value="result", cache_hash="single", store_cubin=False)
        tuner.launchers = [launcher]
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(os.environ, TRITON_CACHE_DIR=directory),
            config.patch(bundle_triton_into_fx_graph_cache=True, use_static_triton_launcher=False),
        ):
            for kernel_hash in ("tuned", "single"):
                kernel = Path(directory) / kernel_hash
                kernel.mkdir()
                (kernel / "kernel.npubin").write_bytes(kernel_hash.encode())
            TritonBundler.begin_compile()
            try:
                TritonBundler.put("tuned", 2)
                TritonBundler.put_winner("tuned")
                self.assertEqual(tuner.run("input", stream=0), "result")
                self.assertEqual(tuner.run("input", stream=0), "result")
                bundle, _ = TritonBundler.collect()
                self.assertEqual(
                    {artifact.kernel_hash for artifact in bundle.kernel_artifacts},
                    {"tuned", "single"},
                )
            finally:
                TritonBundler.end_compile()

    def test_kernel_export_preserves_bundle_registration(self):
        tuner = self.autotuner()
        launcher = mock.Mock(cache_hash="export", store_cubin=True)
        tuner.launchers = [launcher]
        with (
            mock.patch.object(tuner, "save_gpu_kernel") as save,
            mock.patch.object(TritonBundler, "_entries", []),
            mock.patch.object(TritonBundler, "_winners", OrderedSet()),
        ):
            tuner.run(stream=3)
            save.assert_called_once_with(3, launcher)
            self.assertEqual([entry.kernel_hash for entry in TritonBundler._entries], ["export"])
            self.assertIn("export", TritonBundler._winners)

    def test_candidate_generation_uses_producer_configuration(self):
        with (
            ncfg.patch(autotune_enhance=False),
            config.patch({"max_autotune": False, "max_autotune_pointwise": False,
                          "triton.autotune_pointwise": True}),
        ):
            metadata = NPUTritonKernel.inductor_meta_common()
        candidates = []
        for current in (False, True):
            with (
                ncfg.patch(autotune_enhance=current),
                config.patch(max_autotune=current, max_autotune_pointwise=current),
                mock.patch.object(heuristics, "autotune_hints_to_configs", return_value=[]),
                mock.patch.object(heuristics, "cached_autotune") as cached,
            ):
                heuristics.pointwise({"x": 65536}, {"device": object()},
                                     inductor_meta=metadata.copy())
                candidates.append([heuristics._npu_config_key(c) for c in cached.call_args.args[1]])
        self.assertEqual(candidates[0], candidates[1])

    def test_worker_registration_preserves_device_and_cache_directory(self):
        tuner = self.autotuner()
        tuner.compile_results = [SimpleNamespace(kernel=SimpleNamespace(hash="a" * 64))]
        other = self.autotuner()
        other.compile_results = tuner.compile_results
        with mock.patch.object(TritonBundler, "_entries", []):
            for current in (tuner, tuner, other):
                current._precompile_worker()
            self.assertEqual(len(set(TritonBundler._entries)), 1)
            other.triton_meta["device"] = 3
            other._precompile_worker()
            self.assertEqual(len(set(TritonBundler._entries)), 2)
            with mock.patch.dict(os.environ, {"TRITON_CACHE_DIR": "/different/cache"}):
                tuner._precompile_worker()
            self.assertEqual(len(set(TritonBundler._entries)), 3)
        with mock.patch.object(TritonBundler, "_entries", []):
            tuner._precompile_worker()
            self.assertEqual(len(TritonBundler._entries), 1)

    def test_fast_run_registers_reused_kernel_in_a_new_bundle(self):
        tuner = self.autotuner()
        launcher = mock.Mock(return_value="result", cache_hash="single", store_cubin=False)
        tuner.launchers = [launcher]
        cached = mock.Mock(return_value="result", store_cubin=False)
        with mock.patch.object(tuner, "_build_fast_launcher", return_value=cached) as build:
            with (
                mock.patch.object(TritonBundler, "_entries", None),
                mock.patch.object(TritonBundler, "_winners", None),
            ):
                tuner.run("input", stream=0)
            fast_run = tuner.run
            for _ in range(2):
                with (
                    mock.patch.object(TritonBundler, "_entries", []),
                    mock.patch.object(TritonBundler, "_winners", OrderedSet(["tuned"])),
                ):
                    for _ in range(2):
                        self.assertEqual(tuner.run("input", stream=0), "result")
                        self.assertIs(tuner.run, fast_run)
                    tuner.run("input", stream=0, benchmark_run=True)
                    tuner.run("input", stream=0, extra_arg=1)
                    self.assertEqual({e.kernel_hash for e in TritonBundler._entries}, {"single"})
                    self.assertEqual(set(TritonBundler._winners), {"tuned", "single"})
            build.assert_called_once_with(launcher)
            self.assertEqual(cached.call_count, 4)
            replacement = mock.Mock(return_value="replacement", cache_hash="replacement", store_cubin=False)
            tuner.launchers = [replacement]
            with mock.patch.object(TritonBundler, "_entries", None):
                self.assertEqual(tuner.run("input", stream=0), "replacement")

if __name__ == "__main__":
    run_tests()
