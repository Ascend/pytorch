"""AOTI entrypoint tests using CPU PyTorch and mocked backend registration."""

import functools
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch._inductor as inductor


SOURCE = Path(__file__).resolve().parents[2] / "torch_npu" / "utils" / "_dynamo.py"


class MetadataNpuTensor(torch.Tensor):
    @property
    def device(self):
        # Storage stays on CPU; only the metadata inspected by the hook is NPU.
        return types.SimpleNamespace(type="npu")


class TestAotiBackendSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the complete source module without importing the native NPU
        # extension. All hook code and the existing backend scope remain real.
        modules = {}
        for name in (
            "torch_npu", "torch_npu._C", "torch_npu._inductor",
            "torch_npu._inductor.deterministic_cache",
        ):
            modules[name] = types.ModuleType(name)
            modules[name].__path__ = []
        modules["torch_npu"]._C = modules["torch_npu._C"]
        modules["torch_npu._C"]._NPUShapeHandling = type("ShapeHandling", (), {})
        modules["torch_npu._inductor.deterministic_cache"].patch_npu_deterministic_level_cache_keys = mock.Mock()
        cls.module_patch = mock.patch.dict(sys.modules, modules)
        cls.module_patch.start()
        cls.addClassCleanup(cls.module_patch.stop)
        spec = importlib.util.spec_from_file_location("_aoti_dynamo_source_test", SOURCE)
        cls.dynamo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.dynamo)

    def setUp(self):
        self.events = []
        register = mock.patch.object(
            self.dynamo, "register_inductor_npu", side_effect=lambda: self.events.append("register")
        )
        self.register = register.start()
        self.addCleanup(register.stop)
        lazy = mock.patch.object(
            self.dynamo, "_lazy_inductor_setup", side_effect=lambda: self.events.append("setup")
        )
        self.setup = lazy.start()
        self.addCleanup(lazy.stop)
        environment = mock.patch.dict(os.environ, {"TORCHINDUCTOR_NPU_BACKEND": "triton"})
        environment.start()
        self.addCleanup(environment.stop)
        self.npu = torch.Tensor._make_subclass(MetadataNpuTensor, torch.empty(()), False)
        self.options = {"npu_backend": "ascendc", "aot_inductor.precompile_headers": True}
        self.old_pch = inductor.config.aot_inductor.precompile_headers

    def graph(self, *, npu):
        graph = torch.fx.Graph()
        value = graph.placeholder("x")
        value.meta["val"] = self.npu if npu else torch.empty(())
        graph.output(value)
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    def entrypoints(self, callback):
        def recorder(original):
            signature = inspect.signature(original)
            @functools.wraps(original, updated=())
            def call(*args, **kwargs):
                return callback(signature.bind(*args, **kwargs))
            return call
        return types.SimpleNamespace(
            config=inductor.config,
            aoti_compile_and_package=recorder(inductor.aoti_compile_and_package),
            aot_compile=recorder(inductor.aot_compile),
        )

    def test_package_initializes_backend_before_compile_and_restores_state(self):
        for backend, previous_backend in (
            ("ascendc", "triton"), ("triton", "triton"), ("ascendc", None)
        ):
            with self.subTest(backend=backend, previous_backend=previous_backend):
                self.events.clear()
                if previous_backend is None:
                    os.environ.pop("TORCHINDUCTOR_NPU_BACKEND", None)
                else:
                    os.environ["TORCHINDUCTOR_NPU_BACKEND"] = previous_backend
                options = dict(self.options, npu_backend=backend)
                original_options = options.copy()

                def compile_package(bound):
                    self.assertEqual(self.events, ["register", "setup"])
                    self.assertEqual(os.environ["TORCHINDUCTOR_NPU_BACKEND"], backend)
                    self.assertEqual(inductor.config.aot_inductor.precompile_headers, self.old_pch)
                    self.assertEqual(bound.arguments["inductor_configs"], original_options)
                    self.assertEqual(bound.arguments["package_path"], "model.pt2")
                    return "model.pt2"

                module = self.entrypoints(compile_package)
                self.dynamo._patch_aoti_backend_selection(module)
                result = module.aoti_compile_and_package(
                    types.SimpleNamespace(graph_module=self.graph(npu=True)),
                    package_path="model.pt2", inductor_configs=options,
                )
                self.assertEqual(result, "model.pt2")
                self.assertEqual(options, original_options)
                if previous_backend is None:
                    self.assertNotIn("TORCHINDUCTOR_NPU_BACKEND", os.environ)
                else:
                    self.assertEqual(os.environ["TORCHINDUCTOR_NPU_BACKEND"], previous_backend)
                self.assertEqual(inductor.config.aot_inductor.precompile_headers, self.old_pch)

    def test_cpu_graph_does_not_initialize_npu_or_rewrite_options(self):
        def compile_package(bound):
            self.assertEqual(bound.arguments["inductor_configs"], self.options)
            self.assertEqual(os.environ["TORCHINDUCTOR_NPU_BACKEND"], "triton")
        module = self.entrypoints(compile_package)
        self.dynamo._patch_aoti_backend_selection(module)
        module.aoti_compile_and_package(
            types.SimpleNamespace(graph_module=self.graph(npu=False)), inductor_configs=self.options
        )
        self.register.assert_not_called()
        self.setup.assert_not_called()

    def test_aot_compile_detects_npu_example_inputs(self):
        options = dict(self.options)
        original_options = options.copy()

        def compile_graph(bound):
            self.assertEqual(self.events, ["register", "setup"])
            self.assertEqual(os.environ["TORCHINDUCTOR_NPU_BACKEND"], "ascendc")
            self.assertEqual(bound.arguments["options"], original_options)
            return "model.so"

        module = self.entrypoints(compile_graph)
        self.dynamo._patch_aoti_backend_selection(module)
        self.assertEqual(
            module.aot_compile(self.graph(npu=False), (self.npu,), options=options),
            "model.so",
        )
        self.assertEqual(options, original_options)
        self.assertEqual(os.environ["TORCHINDUCTOR_NPU_BACKEND"], "triton")
        self.assertEqual(inductor.config.aot_inductor.precompile_headers, self.old_pch)

    def test_backend_env_restores_and_pch_is_unchanged_on_failure(self):
        for stage in ("registration", "compile"):
            with self.subTest(stage=stage):
                self.register.side_effect = lambda: self.events.append("register")
                callback = mock.Mock()
                module = self.entrypoints(callback)
                self.dynamo._patch_aoti_backend_selection(module)
                failing_call = self.register if stage == "registration" else callback
                error = f"{stage} failed"
                failing_call.side_effect = RuntimeError(error)
                with self.assertRaisesRegex(RuntimeError, error):
                    module.aoti_compile_and_package(
                        self.graph(npu=True), inductor_configs=self.options
                    )
                self.assertEqual(os.environ["TORCHINDUCTOR_NPU_BACKEND"], "triton")
                self.assertEqual(inductor.config.aot_inductor.precompile_headers, self.old_pch)
                if stage == "registration":
                    callback.assert_not_called()

    def test_already_imported_inductor_hook_is_idempotent(self):
        module = self.entrypoints(lambda bound: None)
        with mock.patch.dict(sys.modules, {"torch._inductor": module}):
            self.dynamo._install_aoti_post_import_trigger()
            first = module.aoti_compile_and_package
            self.dynamo._install_aoti_post_import_trigger()
            self.assertIs(module.aoti_compile_and_package, first)
        self.assertTrue(first._torch_npu_aoti_backend_selection)

    def test_post_import_loader_wraps_entries_and_removes_finder(self):
        module = self.entrypoints(lambda bound: None)
        finder = self.dynamo._AotiPostImportFinder()
        loader = mock.Mock()
        hook = self.dynamo._AotiPostImportLoader(loader, finder)
        with mock.patch.object(sys, "meta_path", [finder]):
            hook.exec_module(module)
            self.assertNotIn(finder, sys.meta_path)
        loader.exec_module.assert_called_once_with(module)
        self.assertTrue(module.aoti_compile_and_package._torch_npu_aoti_backend_selection)


if __name__ == "__main__":
    unittest.main()
