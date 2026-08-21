# Owner(s): ["module: dynamo"]

import importlib.util
import os
import subprocess
import sys
import textwrap
import unittest


class TorchCompileTriggerTests(unittest.TestCase):
    def run_in_subprocess(self, code):
        env = os.environ.copy()
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True,
            env=env,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Subprocess failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

    # Verify import installs only the Dynamo post-import trigger.
    def test_import_installs_dynamo_post_import_trigger(self):
        self.run_in_subprocess(
            """
            import sys
            import torch

            def loaded(prefix):
                return [
                    name for name in sys.modules
                    if name == prefix or name.startswith(prefix + ".")
                ]

            assert not loaded("torch._dynamo")
            assert not loaded("torch._inductor")
            src_compile = torch.compile
            src_wrapper_init = torch._TorchCompileWrapper.__init__

            import torch_npu
            from torch_npu.utils import _dynamo

            assert not loaded("torch._dynamo")
            assert not loaded("torch._inductor")
            assert torch.compile is src_compile
            assert torch._TorchCompileWrapper.__init__ is src_wrapper_init
            assert any(
                isinstance(finder, _dynamo._DynamoPostImportFinder)
                for finder in sys.meta_path
            )
            assert not _dynamo._lazy_dynamo_setup.has_run
            """
        )

    # Explicitly importing the NPU Inductor module initializes its config.
    def test_explicit_inductor_import_initializes_npu_config(self):
        self.run_in_subprocess(
            """
            import sys

            import torch
            import torch_npu

            assert "torch._inductor" not in sys.modules
            assert "torch_npu._inductor" not in sys.modules

            import torch_npu._inductor
            from torch._inductor import config

            assert config.npu_backend == "default"
            assert config.enable_shape_handling is False

            config.npu_backend = "mlir"
            assert config.npu_backend == "mlir"
            config.npu_backend = "default"
            """
        )

    # Verify public compiler APIs work before the first compile.
    def test_public_compiler_entries_are_available_before_compile(self):
        self.run_in_subprocess(
            """
            import inspect
            import sys
            import torch
            import torch_npu
            from torch_npu.utils import _dynamo

            marker = torch.compiler.npugraph_mark_step_begin
            assert marker.__name__ == "npugraph_mark_step_begin"
            assert str(inspect.signature(marker)) == "()"
            marker()

            from torch_npu.npu._graph_tree_state import MarkStepBox

            assert MarkStepBox.mark_step_counter == -1
            assert not _dynamo._lazy_dynamo_setup.has_run
            assert not _dynamo._lazy_inductor_setup.has_run
            assert not any(
                name == "torch._dynamo" or name.startswith("torch._dynamo.")
                for name in sys.modules
            )
            assert not any(
                name == "torch._inductor" or name.startswith("torch._inductor.")
                for name in sys.modules
            )

            backends = torch.compiler.list_backends(exclude_tags=None)
            assert {"npu", "npugraph_ex", "npugraphs"}.issubset(backends)
            assert _dynamo._lazy_dynamo_setup.has_run
            assert not _dynamo._lazy_inductor_setup.has_run
            assert "torch_npu._inductor" not in sys.modules
            """
        )

    # Verify wheel metadata exposes loadable NPU backends without compiler imports.
    def test_dynamo_backend_entrypoint_metadata_and_cold_load(self):
        self.run_in_subprocess(
            """
            import importlib.metadata
            import sys

            expected = {
                "npu": "torch_npu.dynamo:_npu_backend_entrypoint",
                "npugraph_ex": (
                    "torch_npu.dynamo:_npugraph_ex_backend_entrypoint"
                ),
                "npugraphs": "torch_npu.dynamo:_npugraphs_backend_entrypoint",
            }
            entry_points = {
                entry_point.name: entry_point
                for entry_point in importlib.metadata.entry_points(
                    group="torch_dynamo_backends"
                )
                if entry_point.name in expected
            }

            assert {
                name: entry_point.value
                for name, entry_point in entry_points.items()
            } == expected
            assert "torch_npu" not in sys.modules
            assert "torch._dynamo" not in sys.modules
            assert "torch._inductor" not in sys.modules

            loaded = {
                name: entry_points[name].load()
                for name in expected
            }
            assert all(callable(backend) for backend in loaded.values())
            assert "torch._dynamo" not in sys.modules
            assert "torch._inductor" not in sys.modules
            """
        )

    # Verify entry-point loading cannot register the NPU backend twice.
    def test_backend_entrypoint_can_import_torch_npu_without_duplicate_registration(self):
        self.run_in_subprocess(
            """
            import torch
            import torch._dynamo
            from torch._dynamo.backends import registry

            # Model the state produced by setuptools entry-point discovery:
            # lookup_backend owns the final register_backend call, while
            # EntryPoint.load imports torch_npu and triggers its lazy setup.
            class NpuEntryPoint:
                module = "torch_npu.dynamo"

                def load(self):
                    from torch_npu.dynamo import _npu_backend_entrypoint
                    return _npu_backend_entrypoint

            registry._BACKENDS["npu"] = NpuEntryPoint()
            backend = registry.lookup_backend("npu")

            assert backend.__name__ == "_npu_backend_entrypoint"
            assert registry._COMPILER_FNS["npu"] is backend

            # A later setup retry must treat the loaded torch_npu entry point
            # as an already completed registration.
            from torch_npu.dynamo import _register_npu_backend

            _register_npu_backend(backend, "npu")
            assert registry._COMPILER_FNS["npu"] is backend
            """
        )

    # Verify backend registration is safe to retry after a partial failure.
    def test_backend_registration_retry_after_partial_failure(self):
        self.run_in_subprocess(
            """
            import torch
            import torch_npu
            import torch._dynamo
            import torch_npu.dynamo as npu_dynamo
            from torch._dynamo.backends import registry

            # Start from an undiscovered state so the first registration is
            # committed before the simulated second-registration failure.
            for name in ("npu", "npugraph_ex"):
                registry._BACKENDS.pop(name, None)
                registry._COMPILER_FNS.pop(name, None)

            original_register = npu_dynamo._register_npu_backend
            fail_npugraph_ex = True

            def fail_second_registration(backend, name="npu"):
                if name == "npugraph_ex" and fail_npugraph_ex:
                    raise RuntimeError("simulated npugraph_ex registration failure")
                return original_register(backend, name)

            npu_dynamo._register_npu_backend = fail_second_registration
            try:
                try:
                    npu_dynamo._register_backends()
                except RuntimeError as error:
                    assert "simulated npugraph_ex" in str(error)
                else:
                    raise AssertionError("the first registration should fail")
            finally:
                npu_dynamo._register_npu_backend = original_register

            assert "npu" in registry._COMPILER_FNS
            assert "npugraph_ex" not in registry._COMPILER_FNS

            # Retry: the completed npu registration is a no-op, while the
            # missing npugraph_ex registration is installed normally.
            npu_dynamo._register_backends()
            assert "npu" in registry._COMPILER_FNS
            assert "npugraph_ex" in registry._COMPILER_FNS
            """
        )

    # Verify failed lazy setup retries through public backend entries.
    def test_public_lazy_setup_entries_retry_after_failure(self):
        self.run_in_subprocess(
            """
            import os
            import sys

            import torch
            import torch_npu
            import torch_npu.dynamo as npu_dynamo
            from torch_npu.utils import _dynamo

            original_patch = _dynamo.patch_dynamo_optimize
            original_get_backend = npu_dynamo._get_default_backend
            attempts = []

            def fail_first_dynamo_setup():
                attempts.append("dynamo")
                if len(attempts) == 1:
                    raise RuntimeError("simulated Dynamo setup failure")
                return original_patch()

            _dynamo.patch_dynamo_optimize = fail_first_dynamo_setup
            npu_dynamo._get_default_backend = (
                lambda name: npu_dynamo._eager_npu_backend
            )
            graph_module = lambda x: x
            try:
                try:
                    npu_dynamo._npu_backend_entrypoint(graph_module, [])
                except RuntimeError as error:
                    assert "simulated Dynamo setup failure" in str(error)
                else:
                    raise AssertionError("the first backend setup should fail")

                assert not _dynamo._lazy_dynamo_setup.has_run
                result = npu_dynamo._npu_backend_entrypoint(graph_module, [])
                assert result is graph_module
            finally:
                _dynamo.patch_dynamo_optimize = original_patch
                npu_dynamo._get_default_backend = original_get_backend

            assert attempts == ["dynamo", "dynamo"]
            assert _dynamo._lazy_dynamo_setup.has_run

            original_register = _dynamo.register_inductor_npu
            original_env = os.environ.get("TORCHINDUCTOR_NPU_BACKEND")
            inductor_attempts = []

            def fail_first_inductor_setup():
                inductor_attempts.append("inductor")
                if len(inductor_attempts) == 1:
                    raise RuntimeError("simulated Inductor setup failure")
                return original_register()

            # Merely creating an Inductor wrapper must not load the backend
            # (issue #4094): the deferred load happens inside the backend scope
            # entered on the first backend execution.
            torch.compile(lambda x: x + 1, backend="inductor")
            assert not _dynamo._lazy_inductor_setup.has_run
            assert "torch_npu._inductor" not in sys.modules

            # Drive the deferred load through the real ``_NpuBackendScope`` used
            # by ``_TorchCompileInductorWrapper.__call__`` (``new_call`` runs
            # ``with _NpuBackendScope(backend): src_call(...)``). The retry
            # semantics live entirely in ``__enter__``, so this exercises the
            # real code path without depending on a full Inductor codegen (the
            # host toolchain cannot always complete one).
            _dynamo.register_inductor_npu = fail_first_inductor_setup
            try:
                try:
                    with _dynamo._NpuBackendScope("default"):
                        pass
                except RuntimeError as error:
                    assert "simulated Inductor setup failure" in str(error)
                else:
                    raise AssertionError("the first Inductor setup should fail")

                assert not _dynamo._lazy_inductor_setup.has_run
                assert os.environ.get("TORCHINDUCTOR_NPU_BACKEND") == original_env

                with _dynamo._NpuBackendScope("default"):
                    pass
            finally:
                _dynamo.register_inductor_npu = original_register

            # The failed backend scope accounts for the first attempt. On the
            # retry, the scope registers once and lazy inductor setup repeats
            # that idempotent registration before marking setup complete.
            assert inductor_attempts == ["inductor", "inductor", "inductor"]
            assert _dynamo._lazy_inductor_setup.has_run
            assert _dynamo.is_inductor_npu_initialized()
            assert os.environ.get("TORCHINDUCTOR_NPU_BACKEND") == original_env
            assert "torch_npu._inductor" in sys.modules
            """
        )

    # Verify a forked child discards inherited in-progress lazy setup state.
    def test_public_lazy_setup_recovers_after_fork(self):
        self.run_in_subprocess(
            """
            import os
            import signal
            import threading

            import torch_npu
            import torch_npu.dynamo as npu_dynamo
            from torch_npu.utils import _dynamo

            parent_pid = os.getpid()
            original_add = _dynamo.add_dynamo_methods_init
            original_get_backend = npu_dynamo._get_default_backend
            entered = threading.Event()
            release = threading.Event()

            def block_parent_setup():
                if os.getpid() == parent_pid:
                    entered.set()
                    assert release.wait(timeout=10)
                return original_add()

            _dynamo.add_dynamo_methods_init = block_parent_setup
            setup_thread = threading.Thread(target=_dynamo._lazy_dynamo_setup)
            setup_thread.start()
            assert entered.wait(timeout=5)

            try:
                child_pid = os.fork()
                if child_pid == 0:
                    def timeout(_signal, _frame):
                        raise TimeoutError("lazy setup hung after fork")

                    signal.signal(signal.SIGALRM, timeout)
                    signal.alarm(5)
                    try:
                        npu_dynamo._get_default_backend = (
                            lambda name: npu_dynamo._eager_npu_backend
                        )
                        graph_module = lambda x: x
                        result = npu_dynamo._npu_backend_entrypoint(
                            graph_module, []
                        )
                        assert result is graph_module
                        signal.alarm(0)
                    except BaseException as error:
                        os.write(
                            2,
                            f"child lazy setup failed: {error}\\n".encode(),
                        )
                        os._exit(1)
                    os._exit(0)

                _, status = os.waitpid(child_pid, 0)
            finally:
                release.set()
                setup_thread.join(timeout=10)
                _dynamo.add_dynamo_methods_init = original_add
                npu_dynamo._get_default_backend = original_get_backend

            assert os.WIFEXITED(status)
            assert os.waitstatus_to_exitcode(status) == 0
            assert not setup_thread.is_alive()
            """
        )

    # Verify NPUGraphs rejects unsupported options in every registration order.
    def test_npugraphs_rejects_options_across_registration_order(self):
        self.run_in_subprocess(
            """
            import contextlib
            from unittest import mock

            import torch
            import torch_npu
            from torch_npu.dynamo import _npugraphs_backend_entrypoint
            from torch_npu.utils import _dynamo, _graph_tree

            gm = object()
            inputs = [object()]
            options = {"npu_backend": "mlir"}

            with mock.patch.object(
                _dynamo, "_lazy_dynamo_setup", lambda: None
            ), mock.patch.object(
                _dynamo, "_lazy_inductor_setup", lambda: None
            ), mock.patch.object(
                _dynamo,
                "_NpuBackendScope",
                lambda backend: contextlib.nullcontext(),
            ), mock.patch.object(
                _graph_tree,
                "npugraphs",
                lambda model, args, **kwargs: "unexpected",
            ):
                for backend in (
                    _npugraphs_backend_entrypoint,
                    _graph_tree.NpugraphsBackend(),
                ):
                    try:
                        backend(gm, inputs, options=options)
                    except TypeError as error:
                        assert "unexpected keyword argument 'options'" in str(error)
                    else:
                        raise AssertionError("npugraphs must reject options")
            """
        )

    # Verify public reset reaches NPUGraphs for every registration order.
    def test_npugraphs_reset_protocol_across_registration_order(self):
        for order in ("cold", "hot"):
            with self.subTest(order=order):
                initialize_inductor = (
                    "torch.compile(lambda x: x + 1, backend='inductor')"
                    if order == "hot"
                    else ""
                )
                self.run_in_subprocess(
                    f"""
                    import sys
                    import types
                    from unittest import mock

                    import torch
                    import torch_npu

                    {initialize_inductor}
                    torch.compile(lambda x: x + 1, backend="npugraphs")

                    from torch._dynamo.backends import registry
                    from torch_npu.dynamo import _npugraphs_backend_entrypoint

                    backend = registry._COMPILER_FNS["npugraphs"]
                    assert backend is _npugraphs_backend_entrypoint
                    assert hasattr(backend, "reset")

                    graph_tree_module = "torch_npu.npu._graph_tree"
                    assert graph_tree_module not in sys.modules
                    backend.reset()
                    assert graph_tree_module not in sys.modules

                    reset_calls = []
                    fake_graph_tree = types.ModuleType(graph_tree_module)
                    fake_graph_tree.reset_npugraph_trees = (
                        lambda: reset_calls.append("reset")
                    )
                    with mock.patch.dict(
                        sys.modules,
                        {{graph_tree_module: fake_graph_tree}},
                    ):
                        torch.compiler.reset()

                    assert reset_calls == ["reset"]
                    """
                )

    # Verify every public Export entry initializes only Dynamo on NPU.
    def test_npu_export_public_entry_and_import_order_matrix(self):
        cases = {
            "module_export": (
                "",
                "exported = torch.export.export(Model(), (x,))",
            ),
            "prebound_export": (
                "from torch.export import export as export_api",
                "exported = export_api(Model(), (x,))",
            ),
            "prebound_export_for_training": (
                "from torch.export import export_for_training as export_api",
                "exported = export_api(Model(), (x,))",
            ),
            "prebound_export_for_inference": (
                "from torch.export import export_for_inference as export_api",
                "exported = export_api(Model(), (x,))",
            ),
        }
        for name, (pre_import, export_call) in cases.items():
            with self.subTest(name=name):
                self.run_in_subprocess(
                    f"""
                    import sys
                    import torch
                    {pre_import}
                    import torch_npu

                    stream = torch.npu.Stream()

                    class Model(torch.nn.Module):
                        def forward(self, x):
                            x.record_stream(stream)
                            return x + 1

                    x = torch.ones(4, device="npu")
                    {export_call}
                    actual = exported.module()(x)

                    from torch_npu.utils import _dynamo

                    torch.testing.assert_close(actual, x + 1)
                    assert _dynamo._lazy_dynamo_setup.has_run
                    assert not _dynamo._lazy_inductor_setup.has_run
                    assert "torch_npu._inductor" not in sys.modules
                    """
                )

    # Verify NPU-specific operations retain their Export capture semantics.
    def test_npu_export_capture_semantics_matrix(self):
        self.run_in_subprocess(
            """
            import sys
            import torch
            import torch_npu

            x = torch.ones(4, device="npu")
            stream = torch.npu.Stream()
            event = torch.npu.Event()

            class StreamAndEvent(torch.nn.Module):
                def forward(self, value):
                    event.record()
                    with torch.npu.stream(stream):
                        event.wait(stream)
                        result = value + 1
                    return result

            class Autocast(torch.nn.Module):
                def forward(self, value):
                    with torch.npu.amp.autocast(dtype=torch.float16):
                        return value * value

            class CurrentDevice(torch.nn.Module):
                def forward(self, value):
                    return value + torch.npu.current_device()

            class DeviceProperties(torch.nn.Module):
                def forward(self, value):
                    properties = torch.npu.get_device_properties(
                        torch.npu.current_device()
                    )
                    return value + 1 if properties.total_memory > 0 else value - 1

            class IsAvailable(torch.nn.Module):
                def forward(self, value):
                    return value + 1 if torch.npu.is_available() else value - 1

            models = (
                StreamAndEvent,
                Autocast,
                CurrentDevice,
                DeviceProperties,
                IsAvailable,
            )
            for model_type in models:
                model = model_type()
                expected = model(x)
                exported = torch.export.export(model, (x,))
                actual = exported.module()(x)
                torch.testing.assert_close(actual, expected)

            from torch_npu.utils import _dynamo

            assert _dynamo._lazy_dynamo_setup.has_run
            assert not _dynamo._lazy_inductor_setup.has_run
            assert "torch_npu._inductor" not in sys.modules
            """
        )

    # Verify rejected Inductor arguments do not initialize or pollute NPU state.
    def test_invalid_inductor_arguments_fail_without_npu_initialization(self):
        cases = {
            "invalid_mode": (
                'torch.compile(lambda x: x + 1, mode="invalid-mode")',
                "Unrecognized mode=invalid-mode",
            ),
            "invalid_option": (
                "torch.compile(lambda x: x + 1, "
                'options={"invalid.option": True})',
                "Unexpected optimization option invalid.option",
            ),
            "invalid_npu_option_type": (
                "torch.compile(lambda x: x + 1, "
                'options={"npu_backend": 1})',
                "Unexpected type of attr npu_backend",
            ),
        }
        for name, (compile_call, expected_error) in cases.items():
            with self.subTest(name=name):
                self.run_in_subprocess(
                    f"""
                    import os
                    import sys

                    import torch
                    import torch_npu
                    from torch_npu.utils import _dynamo

                    env_name = "TORCHINDUCTOR_NPU_BACKEND"
                    original_env = os.environ.get(env_name)
                    try:
                        {compile_call}
                    except RuntimeError as error:
                        assert {expected_error!r} in str(error), str(error)
                    else:
                        raise AssertionError("invalid compile arguments must fail")

                    assert not _dynamo._lazy_inductor_setup.has_run
                    assert not _dynamo.is_inductor_npu_initialized()
                    assert "torch_npu._inductor" not in sys.modules
                    assert os.environ.get(env_name) == original_env
                    """
                )

    # Verify scope entry failures restore the process environment.
    def test_npu_backend_scope_restores_env_after_entry_failure(self):
        self.run_in_subprocess(
            """
            import os

            import torch_npu
            from torch_npu.utils import _dynamo

            env_name = "TORCHINDUCTOR_NPU_BACKEND"
            original_env = os.environ.get(env_name)
            try:
                with _dynamo._NpuBackendScope(1):
                    raise AssertionError("scope entry must fail")
            except TypeError as error:
                assert "str" in str(error)
            else:
                raise AssertionError("a non-string backend must fail")

            assert os.environ.get(env_name) == original_env
            """
        )

    # Verify shape handling is installed before selecting the requested backend.
    def test_shape_handling_initializes_before_backend_selection(self):
        self.run_in_subprocess(
            """
            import os
            from unittest import mock

            import torch
            import torch_npu
            from torch_npu.utils import _dynamo

            options = {
                "npu_backend": "mlir",
                "enable_shape_handling": True,
            }
            events = []

            def scope_register():
                events.append(("scope_register", os.environ["TORCHINDUCTOR_NPU_BACKEND"]))

            with mock.patch.object(
                _dynamo, "_lazy_dynamo_setup", lambda: None
            ), mock.patch.object(
                _dynamo, "_patch_shape_handling",
                lambda: events.append(("shape_handling", None)),
            ), mock.patch.object(
                _dynamo, "_lazy_inductor_setup", lambda: None
            ), mock.patch.object(
                _dynamo, "register_inductor_npu", scope_register
            ):
                wrapper = torch._TorchCompileInductorWrapper(None, options, None)

            assert wrapper.config["npu_backend"] == "mlir"
            assert wrapper.config["enable_shape_handling"] is True
            assert events == [
                ("shape_handling", None),
                ("scope_register", "mlir"),
            ], events
            """
        )

    # Shape Handling must not defeat deferred NPU Inductor loading.
    def test_shape_handling_backend_load_is_deferred_until_first_call(self):
        self.run_in_subprocess(
            """
            import sys

            import torch
            import torch_npu
            from torch_npu.utils import _dynamo

            torch.compile(
                lambda x: x + 1,
                backend="inductor",
                options={"enable_shape_handling": True},
            )

            assert _dynamo._lazy_dynamo_setup.has_run
            assert not _dynamo._lazy_inductor_setup.has_run
            assert "torch_npu._inductor" not in sys.modules
            """
        )
    # Verify non-Inductor compile backends do not initialize Inductor.
    def test_non_inductor_compile_backend_matrix(self):
        cases = {
            "eager": (
                "",
                'compiled = torch.compile(Model(), backend="eager", fullgraph=True)',
            ),
            "custom": (
                "custom_backend = lambda graph_module, example_inputs: "
                "graph_module.forward",
                "compiled = torch.compile(Model(), backend=custom_backend, fullgraph=True)",
            ),
            "npu": (
                "",
                'compiled = torch.compile(Model(), backend="npu", fullgraph=True)',
            ),
        }
        for name, (backend_definition, compile_call) in cases.items():
            with self.subTest(name=name):
                allow_missing_torchair = name == "npu"
                self.run_in_subprocess(
                    f"""
                    import sys
                    import torch
                    import torch_npu

                    class Model(torch.nn.Module):
                        def forward(self, x):
                            return torch.sin(x) + 1

                    {backend_definition}
                    x = torch.randn(8, device="npu")
                    try:
                        {compile_call}
                    except AssertionError as error:
                        assert {allow_missing_torchair!r}
                        assert "Could not find module torchair" in str(error)
                    else:
                        torch.testing.assert_close(compiled(x), Model()(x))

                    from torch_npu.utils import _dynamo

                    assert _dynamo._lazy_dynamo_setup.has_run
                    assert not _dynamo._lazy_inductor_setup.has_run
                    assert "torch_npu._inductor" not in sys.modules
                    """
                )

    # Verify ONNX Dynamo Export initializes only the NPU Dynamo integration.
    @unittest.skipUnless(importlib.util.find_spec("onnxscript"), "requires onnxscript")
    def test_npu_onnx_dynamo_export_initialization_chain(self):
        cases = {
            "module_export": (
                "",
                "result = torch.onnx.export(Model(), (x,), dynamo=True)",
            ),
            "prebound_export": (
                "from torch.onnx import export as onnx_export",
                "result = onnx_export(Model(), (x,), dynamo=True)",
            ),
            "prebound_legacy_dynamo_export": (
                "from torch.onnx import dynamo_export as onnx_export",
                "result = onnx_export(Model(), x)",
            ),
        }
        for name, (pre_import, export_call) in cases.items():
            with self.subTest(name=name):
                self.run_in_subprocess(
                    f"""
                    import sys
                    import torch
                    {pre_import}
                    import torch_npu

                    class Model(torch.nn.Module):
                        def forward(self, x):
                            return torch.sin(x) + 1

                    x = torch.randn(8, device="npu")
                    {export_call}

                    from torch_npu.utils import _dynamo

                    assert result is not None
                    assert _dynamo._lazy_dynamo_setup.has_run
                    assert not _dynamo._lazy_inductor_setup.has_run
                    assert "torch_npu._inductor" not in sys.modules
                    """
                )

    # Verify pre-imported FSDP receives all NPU patches.
    def test_fsdp_patch_when_imported_before_torch_npu(self):
        self.run_in_subprocess(
            """
            import torch.distributed.fsdp
            import torch_npu
            from torch.distributed.fsdp import sharded_grad_scaler
            from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
                FSDPParamGroup,
            )
            from torch_npu.distributed.fsdp._add_fsdp_patch import _patched_finalize_backward
            from torch_npu.npu.amp.sharded_grad_scaler import _ShardedGradScaler

            assert FSDPParamGroup.finalize_backward is _patched_finalize_backward
            assert sharded_grad_scaler.ShardedGradScaler is _ShardedGradScaler
            """
        )

    # Verify a real NPU Inductor compile initializes the full stack.
    @unittest.skipUnless(importlib.util.find_spec("triton"), "requires triton-ascend")
    def test_npu_inductor_initialization_chain(self):
        self.run_in_subprocess(
            """
            import torch
            import torch_npu

            def fn(x):
                return torch.sin(x) + 1

            x = torch.randn(8, device="npu")
            actual = torch.compile(fn, backend="inductor", fullgraph=True)(x)

            from torch._dynamo.device_interface import get_interface_for_device
            from torch_npu.utils import _dynamo

            torch.testing.assert_close(actual, fn(x))
            assert _dynamo._lazy_dynamo_setup.has_run
            assert _dynamo._lazy_inductor_setup.has_run
            assert get_interface_for_device("npu").device_count() > 0
            """
        )

    # Issue #4094: creating an Inductor wrapper must not import torch_npu._inductor.
    def test_inductor_backend_load_is_deferred_until_first_call(self):
        self.run_in_subprocess(
            """
            import sys
            import torch
            import torch_npu
            from torch_npu.utils import _dynamo

            torch.compile(
                lambda x: x + 1,
                backend="inductor",
                options={"enable_shape_handling": True},
            )

            assert _dynamo._lazy_dynamo_setup.has_run
            assert not _dynamo._lazy_inductor_setup.has_run
            assert "torch_npu._inductor" not in sys.modules
            """
        )

    # Issue #4204: Shape Handling must install the Dynamo hook at wrapper
    # creation (before the Dynamo context is built) while still deferring the
    # heavyweight NPU Inductor/Triton load to the first backend execution.
    def test_shape_handling_hook_installed_before_backend_load(self):
        self.run_in_subprocess(
            """
            import sys
            import torch
            import torch_npu
            from torch_npu.utils import _dynamo

            events = []
            original_patch = _dynamo._patch_shape_handling
            original_lazy = _dynamo._lazy_inductor_setup

            def track_patch():
                events.append("shape_handling")
                return original_patch()

            def track_lazy():
                events.append("inductor_load")
                return original_lazy()

            _dynamo._patch_shape_handling = track_patch
            _dynamo._lazy_inductor_setup = track_lazy
            try:
                torch.compile(
                    lambda x: x + 1,
                    backend="inductor",
                    options={"enable_shape_handling": True},
                )
                # The lightweight hook is installed at wrapper creation; the
                # full backend is not loaded yet.
                assert events == ["shape_handling"], events
                assert "torch_npu._inductor" not in sys.modules
            finally:
                _dynamo._patch_shape_handling = original_patch
                _dynamo._lazy_inductor_setup = original_lazy
            """
        )

    # Verify a real NPUGraphs compile initializes the full stack.
    @unittest.skipUnless(importlib.util.find_spec("triton"), "requires triton-ascend")
    def test_npu_npugraphs_initialization_chain(self):
        self.run_in_subprocess(
            """
            import torch
            import torch_npu

            def fn(x):
                return torch.sin(x) + 1

            x = torch.randn(8, device="npu")
            actual = torch.compile(fn, backend="npugraphs", fullgraph=True)(x)

            from torch._dynamo.device_interface import get_interface_for_device
            from torch_npu.utils import _dynamo

            torch.testing.assert_close(actual, fn(x))
            assert _dynamo._lazy_dynamo_setup.has_run
            assert _dynamo._lazy_inductor_setup.has_run
            assert get_interface_for_device("npu").device_count() > 0
            """
        )

    # Verify lazy setup completes before compile backend lookup.
    def test_compile_triggers_setup_before_backend_lookup(self):
        self.run_in_subprocess(
            """
            import torch
            import torch_npu
            from torch_npu.utils import _dynamo

            calls = []

            @_dynamo.run_once
            def fake_setup():
                calls.append("setup")

            _dynamo._lazy_dynamo_setup = fake_setup
            from torch._dynamo.backends import registry
            assert fake_setup.has_run

            compiled = torch.compile(lambda x: x + 1, backend="eager")
            assert compiled(torch.tensor(1)).item() == 2
            assert calls == ["setup"]
            """
        )

    # Verify the trigger works when Dynamo was imported first.
    def test_trigger_after_dynamo_was_preimported(self):
        self.run_in_subprocess(
            """
            import sys
            import torch
            import torch._dynamo
            import torch_npu
            from torch_npu.utils import _dynamo

            assert _dynamo._lazy_dynamo_setup.has_run

            compiled = torch.compile(lambda x: x + 1, backend="eager", fullgraph=True)
            assert compiled(torch.tensor(1)).item() == 2
            assert _dynamo._lazy_dynamo_setup.has_run
            assert not _dynamo._lazy_inductor_setup.has_run
            assert "torch_npu._inductor" not in sys.modules
            """
        )

    # Verify all legacy Dynamo patches remain installed exactly once.
    def test_dynamo_patch_inventory_is_preserved(self):
        self.run_in_subprocess(
            """
            import torch
            import torch_npu

            # Importing the Dynamo parent package is the lazy setup boundary.
            import torch._dynamo

            from torch._dynamo.device_interface import get_interface_for_device
            from torch._dynamo.variables.builtin import BuiltinVariable
            from torch._dynamo.variables.builder import VariableBuilder
            from torch._dynamo.variables.ctx_manager import EventVariable
            from torch._dynamo.variables.functions import SkipFunctionVariable
            from torch._dynamo.variables.tensor import TensorVariable
            from torch._dynamo.variables.torch import constant_fold_functions
            from torch._dynamo.variables.user_defined import UserDefinedClassVariable
            from torch._dynamo.utils import common_constant_types
            from torch_npu.dynamo.trace_rule import (
                skip_functions_npu,
                torch_c_binding_in_graph_functions_npu,
                torch_non_c_binding_in_graph_functions_npu,
            )
            from torch_npu.utils import _dynamo

            assert _dynamo._lazy_dynamo_setup.has_run
            assert get_interface_for_device("npu").device_count() > 0

            # VariableTracker and context-manager patches formerly installed
            # eagerly by add_dynamo_methods().
            assert SkipFunctionVariable.__new__.__module__ == "torch_npu.utils._dynamo"
            assert TensorVariable.call_method.__module__ == "torch_npu.utils._dynamo"
            assert UserDefinedClassVariable.__new__.__module__ == "torch_npu.utils._dynamo"
            in_graph_classes = UserDefinedClassVariable._in_graph_classes()
            assert torch.npu.Event in in_graph_classes
            assert torch.npu.Stream in in_graph_classes
            assert torch.npu.fake_record_stream is _dynamo.fake_record_stream
            assert TensorVariable.method_record_stream.__module__ == "torch_npu.utils._dynamo"
            assert VariableBuilder._wrap.__module__ == "torch_npu.utils._dynamo"
            assert BuiltinVariable.call_id.__module__ == "torch_npu.utils._dynamo"
            assert EventVariable.python_type.__module__ == "torch_npu.utils._dynamo"
            assert torch._dynamo.optimize.__module__ == "torch_npu.utils._dynamo"

            # Backend and trace-rule registrations formerly performed by
            # registry_manager._register_dynamo().  Count the maps as well as
            # checking membership so repeated lazy triggers cannot hide a
            # duplicate installation.
            assert {"npu", "npugraph_ex"}.issubset(
                torch._dynamo.list_backends(exclude_tags=None)
            )
            maps = torch._dynamo.trace_rules.torch_name_rule_map
            assert maps.count(torch_non_c_binding_in_graph_functions_npu) == 1
            assert maps.count(torch_c_binding_in_graph_functions_npu) == 1
            assert maps.count(skip_functions_npu) == 1
            assert constant_fold_functions[torch.npu.current_device]
            assert constant_fold_functions[torch.npu.get_device_properties]
            assert constant_fold_functions[torch.npu.is_available]
            assert torch_npu._C._NPUDeviceProperties in common_constant_types
            """
        )

    # Verify all legacy Inductor patches remain installed.
    def test_inductor_patch_inventory_is_preserved(self):
        self.run_in_subprocess(
            """
            import torch
            import torch_npu

            # RNG/decomposition patches remain installed at import time.
            from torch_npu.utils import _inductor

            assert (
                torch._decomp.decompositions._max_unpoolnd
                is _inductor._max_unpoolnd_patch
            )
            assert torch._prims.rng_prims.philox_rand_offset.__module__ == (
                "torch_npu.utils._inductor"
            )
            assert torch._prims.rng_prims.register_philox_rand.__module__ == (
                "torch_npu.utils._inductor"
            )
            assert torch._prims.rng_prims.get_device.__module__ == (
                "torch_npu.utils._inductor"
            )

            # Exercise the new full-Inductor setup boundary without relying on
            # test ordering or on a prior torch.compile invocation.
            import torch._dynamo
            from torch_npu.utils import _dynamo

            _dynamo._lazy_inductor_setup()

            import torch._inductor.compile_fx as compile_fx
            import torch._inductor.cudagraph_trees as cudagraph_trees
            import torch._inductor.cudagraph_utils as cudagraph_utils
            from torch._inductor.codegen.common import get_device_op_overrides
            from torch._inductor.codecache import CacheBase
            from torch._inductor.graph import GraphLowering
            from torch._inductor.utils import GPU_TYPES
            from torch_npu.utils import _graph_tree

            assert _dynamo._lazy_inductor_setup.has_run
            assert "npu" in GPU_TYPES
            assert get_device_op_overrides("npu").__class__.__module__.startswith(
                "torch_npu._inductor"
            )
            assert torch.utils._triton.has_triton.__module__ == (
                "torch_npu._inductor.utils"
            )
            assert torch.utils._triton.has_triton_tma.__module__ == (
                "torch_npu._inductor.utils"
            )
            assert compile_fx.has_triton is torch.utils._triton.has_triton
            assert GraphLowering.codegen_with_cpp_wrapper.__module__ == (
                "torch_npu._inductor.graph"
            )
            assert CacheBase.get_system.__module__ == (
                "torch_npu._inductor.codegen.common"
            )

            # NPUGraph integrations were formerly applied eagerly alongside
            # the Inductor patches.
            assert compile_fx.cudagraphify is _graph_tree.npugraphify
            assert (
                cudagraph_utils.check_multiple_devices_or_any_cpu_nodes
                is _graph_tree.check_multiple_devices_or_any_cpu_nodes
            )
            assert cudagraph_trees.get_manager.__module__ == (
                "torch_npu.utils._graph_tree"
            )
            assert torch.compiler.npugraph_mark_step_begin is (
                _graph_tree.npugraph_mark_step_begin
            )

            config = torch._inductor.config
            assert config.npu_backend == "default"
            assert config.enable_shape_handling is False
            assert config.shape_handling_configs == []
            assert config.shape_handling_dict is None
            """
        )


if __name__ == "__main__":
    unittest.main()
