import ast
import importlib.metadata as importlib_metadata
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
    wanted = {"_parse_bool_env", "_parse_int_env"}
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
        parse_int=ns["_parse_int_env"],
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

    def test_parse_int_env(self):
        mod = load_parse_helpers()
        key = "INDUCTOR_ASCEND_COSTMODEL_TOPK"

        os.environ.pop(key, None)
        self.assertEqual(mod.parse_int(key, default=150, min_value=1), 150)

        os.environ[key] = "8"
        self.assertEqual(mod.parse_int(key, default=150, min_value=1), 8)

        os.environ[key] = "0"
        self.assertEqual(mod.parse_int(key, default=150, min_value=1), 150)

        os.environ[key] = "bad"
        self.assertEqual(mod.parse_int(key, default=150, min_value=1), 150)


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


class TestCostmodelTopkPrecompileOrder(unittest.TestCase):
    def test_prefilter_called_before_precompile(self):
        method_src = _get_class_method_source(
            TRITON_HEURISTICS_PATH,
            "NPUCachingAutotuner",
            "precompile",
        )
        pos_prefilter = method_src.find("self._apply_costmodel_topk_to_configs()")
        pos_warm = method_src.find("if warm_cache_only:")
        self.assertGreaterEqual(pos_prefilter, 0, "missing config prefilter call")
        self.assertGreaterEqual(pos_warm, 0, "missing warm_cache_only branch")
        self.assertLess(pos_prefilter, pos_warm, "prefilter should be at precompile entry")

    def test_costmodel_prefilter_uses_ascend_backend_entry(self):
        src = TRITON_HEURISTICS_PATH.read_text(encoding="utf-8")
        self.assertIn("from triton.backends.ascend.runtime.costmodel_runtime import costmodel_bench", src)


class TestCostmodelBackendIntegration(unittest.TestCase):
    @staticmethod
    def _parse_semver_prefix(ver: str):
        # Accept forms like: 3.2.2, 3.2.2+gitxxxx, 3.2.2.post1
        raw = (ver or "").strip().split("+", 1)[0]
        parts = raw.split(".")
        nums = []
        for item in parts:
            digits = "".join(ch for ch in item if ch.isdigit())
            nums.append(int(digits) if digits else 0)
            if len(nums) >= 3:
                break
        while len(nums) < 3:
            nums.append(0)
        return tuple(nums[:3])

    def test_inductor_can_call_triton_ascend_costmodel_backend(self):
        min_required = (3, 2, 2)
        try:
            installed_ver = importlib_metadata.version("triton-ascend")
        except Exception as exc:
            self.skipTest(f"triton-ascend not installed: {exc}")

        if self._parse_semver_prefix(installed_ver) < min_required:
            self.skipTest(
                f"triton-ascend>={'.'.join(map(str, min_required))} required, got {installed_ver}"
            )

        try:
            from triton.backends.ascend.runtime import costmodel_runtime
            from triton._C.libtriton import ascend as ascend_capi
        except Exception as exc:
            self.skipTest(f"triton-ascend costmodel backend unavailable: {exc}")

        if not hasattr(ascend_capi, "run_costmodel_inproc"):
            self.skipTest("run_costmodel_inproc is not exported by current triton-ascend build")

        mlir_text = "module { func.func @costmodel_smoke() { return } }"
        out = costmodel_runtime.run_costmodel(
            mlir_text,
            extra_args=["-ascend-perf-model"],
        )

        self.assertIsNotNone(out, "costmodel backend should return simulator output text")
        self.assertIsInstance(out, str)
        self.assertTrue(out.strip(), "costmodel backend output should be non-empty")
        self.assertIn("Pipeline Analysis", out)


if __name__ == "__main__":
    unittest.main()
