import importlib
import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest import mock


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
PACKAGE = "torch_npu._inductor.experimental.python_wrapper_fast_launch"


@contextmanager
def isolated_patch_imports(*, fast_launch=False):
    for name in list(sys.modules):
        if name == PACKAGE or name.startswith(PACKAGE + "."):
            sys.modules.pop(name, None)

    torch_npu = types.ModuleType("torch_npu")
    torch_npu.__path__ = [os.path.join(REPO_ROOT, "torch_npu")]
    inductor = types.ModuleType("torch_npu._inductor")
    inductor.__path__ = [os.path.join(REPO_ROOT, "torch_npu", "_inductor")]
    config = types.ModuleType("torch_npu._inductor.config")
    config.enable_fast_launch = fast_launch
    inductor.config = config

    with mock.patch.dict(
        sys.modules,
        {
            "torch_npu": torch_npu,
            "torch_npu._inductor": inductor,
            "torch_npu._inductor.config": config,
        },
    ):
        try:
            yield
        finally:
            for name in list(sys.modules):
                if name == PACKAGE or name.startswith(PACKAGE + "."):
                    sys.modules.pop(name, None)


class _DebugPrinter:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def set_printer_args(self, *args):
        self.args = args


def _fake_wrapper_module():
    events = []

    class FakeWrapper:
        def __init__(self):
            self.imports = types.SimpleNamespace(
                writeline=lambda line: events.append(("import", line))
            )
            self.kernel_autotune_names = set()

        def write_triton_header_once(self):
            events.append(("header",))
            return "header"

        def prepare_triton_kernel_call(self, call_args):
            return tuple(call_args)

        def generate_kernel_call(self, *args, **kwargs):
            events.append(("original", args, kwargs))
            return "original"

    graph = types.SimpleNamespace(
        cpp_wrapper=False,
        get_current_device_or_throw=lambda: types.SimpleNamespace(index=0),
        wrapper_code=types.SimpleNamespace(debug_printer=_DebugPrinter()),
    )
    module = types.SimpleNamespace(
        NPUPythonWrapperCodeGen=FakeWrapper,
        PythonWrapperCodegen=types.SimpleNamespace(
            write_get_raw_stream=lambda owner, index, graph: f"stream{index}"
        ),
        V=types.SimpleNamespace(graph=graph),
        config=types.SimpleNamespace(
            triton=types.SimpleNamespace(autotune_at_compile_time=False)
        ),
        is_multi_stream=lambda: False,
        _is_codegen_graph_partition_subgraph=lambda owner: False,
    )
    return module, FakeWrapper, events


def _fake_triton_module():
    class FakeGrid:
        prefix = ()
        x_grid = "xnumel"
        y_grid = "1"
        z_grid = "1"

    class FakeGridExpr:
        @staticmethod
        def from_meta(inductor_meta, cfg):
            return FakeGrid()

    class FakeCompileResult:
        def __init__(self):
            self.config = types.SimpleNamespace(kwargs={})
            self.compile_meta = {
                "constants": {},
                "signature": {"xnumel": "i64"},
            }
            fn = types.SimpleNamespace(
                __name__="triton_poi_fused_0",
                arg_names=("xnumel",),
                constexprs=set(),
            )
            self.kernel = types.SimpleNamespace(
                src=types.SimpleNamespace(fn=fn),
                metadata=types.SimpleNamespace(),
            )
            self.inductor_meta = {"kernel_name": "triton_poi_fused_0"}

        def make_launcher(self):
            scope = {
                "function": 1234,
                "launch_enter_hook": None,
                "launch_exit_hook": None,
            }
            exec(
                "def launcher(xnumel, stream):\n    return (xnumel, stream)\n",
                scope,
            )
            return scope["launcher"]

    module = types.SimpleNamespace(
        TritonCompileResultNpu=FakeCompileResult,
        OrderedSet=set,
        triton_version_uses_attrs_dict=lambda: False,
        config_to_dict=lambda cfg: dict(cfg.kwargs),
        filter_launcher_def_args=lambda names, cfg, none, runtime: [
            name for name in names if name not in cfg and name not in none
        ],
        GridExpr=FakeGridExpr,
        GridExprNpu=types.SimpleNamespace(),
    )
    return module, FakeCompileResult


class TestNPUFastLaunchPatch(unittest.TestCase):
    def test_default_off_does_not_resolve_patch_targets(self):
        with isolated_patch_imports():
            fast_patch = importlib.import_module(f"{PACKAGE}.patch")
            fast_patch._resolve_wrapper_module = mock.Mock(side_effect=AssertionError)
            fast_patch._resolve_triton_heuristics_module = mock.Mock(
                side_effect=AssertionError
            )

            self.assertFalse(fast_patch.patch_fast_launch())

            fast_patch._resolve_wrapper_module.assert_not_called()
            fast_patch._resolve_triton_heuristics_module.assert_not_called()

    def test_enabled_patch_installs_once_and_routes_codegen(self):
        wrapper_module, wrapper_cls, events = _fake_wrapper_module()
        triton_module, triton_cls = _fake_triton_module()
        emitted = []

        class FakeEmitter:
            def __init__(self, owner):
                self.owner = owner

            def emit_triton_call(self, **kwargs):
                emitted.append(kwargs)

        attach_metadata = mock.Mock()
        with (
            isolated_patch_imports(fast_launch=True),
            mock.patch.dict(
                sys.modules,
                {
                    f"{PACKAGE}.wrapper_codegen": types.SimpleNamespace(
                        FastLaunchWrapperEmitter=FakeEmitter
                    ),
                    f"{PACKAGE}.launcher": types.SimpleNamespace(
                        attach_python_wrapper_launcher_metadata=attach_metadata
                    ),
                },
            ),
        ):
            fast_patch = importlib.import_module(f"{PACKAGE}.patch")
            fast_patch._resolve_wrapper_module = mock.Mock(return_value=wrapper_module)
            fast_patch._resolve_triton_heuristics_module = mock.Mock(
                return_value=triton_module
            )
            original_generate = wrapper_cls.generate_kernel_call
            original_make_launcher = triton_cls.make_launcher

            self.assertTrue(fast_patch.patch_fast_launch())
            self.assertTrue(fast_patch.patch_fast_launch())

            self.assertIsNot(wrapper_cls.generate_kernel_call, original_generate)
            self.assertIsNot(triton_cls.make_launcher, original_make_launcher)
            self.assertEqual(fast_patch._resolve_wrapper_module.call_count, 1)
            self.assertEqual(
                fast_patch._resolve_triton_heuristics_module.call_count,
                1,
            )

            wrapper = wrapper_cls()
            wrapper.generate_kernel_call(
                "triton_poi_fused_0",
                ("xnumel",),
                triton_meta={"signature": {"xnumel": "i64"}},
            )
            self.assertEqual(len(emitted), 1)
            self.assertIn(
                "python_wrapper_fast_launch.bind",
                events[1][1],
            )

            wrapper_module.config.triton.autotune_at_compile_time = True
            self.assertEqual(
                wrapper.generate_kernel_call(
                    "triton_poi_fused_compile_time_autotune",
                    ("buf0",),
                    arg_types=(object(),),
                    raw_args=None,
                ),
                "original",
            )
            self.assertEqual(len(emitted), 1)
            self.assertEqual(events[-1][0], "original")
            wrapper_module.config.triton.autotune_at_compile_time = False

            wrapper_module.V.graph.cpp_wrapper = True
            self.assertEqual(
                wrapper.generate_kernel_call(
                    "triton_poi_fused_1",
                    ("xnumel",),
                ),
                "original",
            )

            launcher = triton_cls().make_launcher()
            attach_metadata.assert_called_once()
            get_grid = attach_metadata.call_args.kwargs["get_grid"]
            self.assertEqual(get_grid(7), (7, 1, 1))
            self.assertEqual(launcher(3, "stream"), (3, "stream"))

            grouped_result = triton_cls()
            grouped_result.inductor_meta["group_enabled"] = True
            grouped_launcher = grouped_result.make_launcher()
            attach_metadata.assert_called_once()
            self.assertEqual(grouped_launcher(5, "stream"), (5, "stream"))

    def test_patch_operation_failure_rolls_back_previous_changes(self):
        class RejectingMeta(type):
            def __setattr__(cls, name, value):
                if name == "reject":
                    raise RuntimeError("reject")
                return super().__setattr__(name, value)

        class Target(metaclass=RejectingMeta):
            first = "original"

        with isolated_patch_imports():
            fast_patch = importlib.import_module(f"{PACKAGE}.patch")
            with self.assertRaisesRegex(RuntimeError, "reject"):
                fast_patch._apply_patch_operations(
                    [
                        (Target, "first", "patched"),
                        (Target, "reject", True),
                    ]
                )
            self.assertEqual(Target.first, "original")
            self.assertFalse(hasattr(Target, "reject"))


if __name__ == "__main__":
    unittest.main()
