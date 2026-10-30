import ast
import os
import pathlib
import types
import unittest


CONFIG_PATH = pathlib.Path(__file__).resolve().parents[2] / "torch_npu" / "_inductor" / "config.py"
TRITON_HEURISTICS_PATH = pathlib.Path(__file__).resolve().parents[2] / "torch_npu" / "_inductor" / "runtime" / "triton_heuristics.py"


class _DummyLog:
    def warning(self, *args, **kwargs):
        return None


def load_parse_helpers():
    src = CONFIG_PATH.read_text(encoding="utf-8")
    module_ast = ast.parse(src)
    wanted = {"_parse_bool_env", "_parse_float_env"}
    func_nodes = [n for n in module_ast.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if len(func_nodes) != 2:
        raise RuntimeError("Failed to locate parser helpers in config.py")

    mini_mod = ast.Module(body=func_nodes, type_ignores=[])
    ast.fix_missing_locations(mini_mod)
    code = compile(mini_mod, str(CONFIG_PATH), "exec")

    ns = {"os": os, "log": _DummyLog()}
    exec(code, ns)
    return types.SimpleNamespace(
        parse_bool=ns["_parse_bool_env"],
        parse_float=ns["_parse_float_env"],
    )


class TestTritonCostmodelControl(unittest.TestCase):
    def test_parse_bool_env(self):
        mod = load_parse_helpers()
        key = "INDUCTOR_ASCEND_ENABLE_COSTMODEL"

        os.environ.pop(key, None)
        self.assertFalse(mod.parse_bool(key, False))
        self.assertTrue(mod.parse_bool(key, True))

        os.environ[key] = "1"
        self.assertTrue(mod.parse_bool(key, False))
        os.environ[key] = "true"
        self.assertTrue(mod.parse_bool(key, False))
        os.environ[key] = "on"
        self.assertTrue(mod.parse_bool(key, False))
        os.environ[key] = "0"
        self.assertFalse(mod.parse_bool(key, True))

    def test_parse_float_env(self):
        mod = load_parse_helpers()
        key = "INDUCTOR_ASCEND_COSTMODEL_RATIO"

        os.environ.pop(key, None)
        self.assertEqual(mod.parse_float(key, default=0.25, min_value=0.0, max_value=1.0), 0.25)

        os.environ[key] = "0.5"
        self.assertEqual(mod.parse_float(key, default=0.25, min_value=0.0, max_value=1.0), 0.5)

        os.environ[key] = "0"
        self.assertEqual(mod.parse_float(key, default=0.25, min_value=0.0, max_value=1.0), 0.25)

        os.environ[key] = "bad"
        self.assertEqual(mod.parse_float(key, default=0.25, min_value=0.0, max_value=1.0), 0.25)


def _get_class_method_source(path: pathlib.Path, class_name: str, method_name: str) -> str:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == method_name:
                    lines = src.splitlines()
                    return "\n".join(lines[sub.lineno - 1: sub.end_lineno])
    raise RuntimeError(f"Method {class_name}.{method_name} not found")


def _load_autotuner_costmodel_helpers():
    src = TRITON_HEURISTICS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)

    wanted = {
        "_try_parse_int_like",
        "_build_ttir_arg_value_map",
        "_build_costmodel_arg_bindings",
    }

    class_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "NPUCachingAutotuner":
            class_node = node
            break
    if class_node is None:
        raise RuntimeError("NPUCachingAutotuner not found in triton_heuristics.py")

    func_nodes = [n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if len(func_nodes) != len(wanted):
        missing = wanted - {n.name for n in func_nodes}
        raise RuntimeError(f"Missing helper methods in triton_heuristics.py: {sorted(missing)}")

    mini_mod = ast.Module(body=func_nodes, type_ignores=[])
    ast.fix_missing_locations(mini_mod)
    code = compile(mini_mod, str(TRITON_HEURISTICS_PATH), "exec")

    ns = {"re": __import__("re")}
    exec(code, ns)
    return types.SimpleNamespace(**{name: ns[name] for name in wanted})


class TestCostmodelBindingHelpers(unittest.TestCase):
    def test_arg_bindings_support_kwargs_and_argid_mapping(self):
        helpers = _load_autotuner_costmodel_helpers()

        ttir_text = """
module {
  tt.func public @softmax_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) attributes {noinline = false} {
    tt.return
  }
}
"""

        dummy = types.SimpleNamespace(
            triton_meta={"signature": {"output_ptr": "*fp32", "input_ptr": "*fp32", "input_row_stride": "i32", "output_row_stride": "i32", "n_rows": "i32", "n_cols": "i32"}}
        )
        dummy._try_parse_int_like = types.MethodType(helpers._try_parse_int_like, dummy)
        dummy._build_ttir_arg_value_map = types.MethodType(helpers._build_ttir_arg_value_map, dummy)

        runtime_args = [object(), object()]
        runtime_kwargs = {
            "input_row_stride": 781,
            "output_row_stride": 781,
            "n_rows": 1823,
            "n_cols": 781,
        }

        bindings = helpers._build_costmodel_arg_bindings(dummy, ttir_text, runtime_args, runtime_kwargs)
        self.assertIn("arg2=781", bindings)
        self.assertIn("arg3=781", bindings)
        self.assertIn("arg4=1823", bindings)
        self.assertIn("arg5=781", bindings)
        self.assertTrue(bindings.endswith("pid_x=0"))


class TestCostmodelRatioPrecompileOrder(unittest.TestCase):
    def test_prefilter_called_before_precompile(self):
        method_src = _get_class_method_source(
            TRITON_HEURISTICS_PATH,
            "NPUCachingAutotuner",
            "precompile",
        )
        pos_prefilter = method_src.find("self._apply_costmodel_to_configs(*runtime_args, **runtime_kwargs)")
        pos_warm = method_src.find("if warm_cache_only:")
        self.assertGreaterEqual(pos_prefilter, 0, "missing config prefilter call")
        self.assertGreaterEqual(pos_warm, 0, "missing warm_cache_only branch")
        self.assertLess(pos_prefilter, pos_warm, "prefilter should be at precompile entry")

    def test_costmodel_prefilter_uses_ascend_backend_entry(self):
        src = TRITON_HEURISTICS_PATH.read_text(encoding="utf-8")
        self.assertIn("from triton.backends.ascend.runtime.costmodel_runtime import costmodel_bench", src)


if __name__ == "__main__":
    unittest.main()
