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


class TestCostmodelBackendIntegration(unittest.TestCase):
    @staticmethod
    def _parse_semver_prefix(ver: str):
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

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        min_required = (3, 2, 2)
        try:
            installed_ver = importlib_metadata.version("triton-ascend")
        except Exception as exc:
            raise unittest.SkipTest(f"triton-ascend not installed: {exc}")
        if cls._parse_semver_prefix(installed_ver) < min_required:
            raise unittest.SkipTest(f"triton-ascend>={'.'.join(map(str, min_required))} required, got {installed_ver}")
        os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
        try:
            import triton  # noqa: F401
            import triton.language as tl  # noqa: F401
            from triton.backends.ascend.runtime import costmodel_runtime
            from triton._C.libtriton import ascend as ascend_capi
            if not hasattr(ascend_capi, "run_costmodel_inproc"):
                raise unittest.SkipTest("run_costmodel_inproc is not exported by current triton-ascend build")
            cls._costmodel_runtime = costmodel_runtime
            return
        except unittest.SkipTest:
            raise
        except Exception:
            dist = importlib_metadata.distribution("triton-ascend")
            cm_path = None
            for f in (dist.files or []):
                if str(f).endswith("triton/backends/ascend/runtime/costmodel_runtime.py"):
                    cm_path = pathlib.Path(dist.locate_file(f))
                    break
            if cm_path is None or not cm_path.exists():
                raise unittest.SkipTest("costmodel_runtime.py not found from installed triton-ascend")
            spec = __import__("importlib.util").util.spec_from_file_location("cm_runtime_test", cm_path)
            mod = __import__("importlib.util").util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)
            cls._costmodel_runtime = mod

    @staticmethod
    def _load_pytorch_costmodel_methods():
        src = TRITON_HEURISTICS_PATH.read_text(encoding="utf-8")
        tree = ast.parse(src)
        wanted = {"_make_ttir_module_from_cfg", "_try_parse_int_like", "_build_ttir_arg_value_map", "_build_costmodel_arg_bindings", "_build_costmodel_items"}
        class_node = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "NPUCachingAutotuner":
                class_node = node
                break
        if class_node is None:
            raise RuntimeError("NPUCachingAutotuner not found")
        func_nodes = [n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        got = {n.name for n in func_nodes}
        if got != wanted:
            raise RuntimeError(f"missing methods: {sorted(wanted - got)}")
        mini_mod = ast.Module(body=func_nodes, type_ignores=[])
        ast.fix_missing_locations(mini_mod)
        code = compile(mini_mod, str(TRITON_HEURISTICS_PATH), "exec")
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        class _NPUKernelTypeMock:
            class _SIMTOnly:
                @staticmethod
                def compile_mode():
                    return "__SIMT_ONLY__"
            SIMT_ONLY = _SIMTOnly()
        class _NPUConfigMock:
            simt_default_warp_stacksize = 8192
        ns = {"copy": __import__("copy"), "os": __import__("os"), "re": __import__("re"), "ASTSource": ASTSource, "GPUTarget": GPUTarget, "NPUKernelType": _NPUKernelTypeMock, "npu_config": _NPUConfigMock}
        exec(code, ns)
        return types.SimpleNamespace(**{name: ns[name] for name in wanted})

    @staticmethod
    def _build_embedded_kernels():
        import importlib.util
        import tempfile
        import textwrap

        src = textwrap.dedent("""
            import triton
            import triton.language as tl

            @triton.jit
            def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(axis=0)
                block_start = pid * BLOCK_SIZE
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = tl.load(y_ptr + offsets, mask=mask)
                output = x + y
                tl.store(output_ptr + offsets, output, mask=mask)

            @triton.jit
            def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
                row_start = tl.program_id(0)
                row_step = tl.num_programs(0)
                for row_idx in tl.range(row_start, n_rows, row_step):
                    row_start_ptr = input_ptr + row_idx * input_row_stride
                    col_offsets = tl.arange(0, BLOCK_SIZE)
                    input_ptrs = row_start_ptr + col_offsets
                    mask = col_offsets < n_cols
                    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
                    row_minus_max = row - tl.max(row, axis=0)
                    numerator = tl.exp(row_minus_max)
                    denominator = tl.sum(numerator, axis=0)
                    softmax_output = numerator / denominator
                    output_row_start_ptr = output_ptr + row_idx * output_row_stride
                    output_ptrs = output_row_start_ptr + col_offsets
                    tl.store(output_ptrs, softmax_output, mask=mask)
        """)

        with tempfile.NamedTemporaryFile(mode="w", suffix="_embedded_tutorial_kernels.py", delete=False) as f:
            f.write(src)
            tmp_path = f.name

        spec = importlib.util.spec_from_file_location("embedded_tutorial_kernels", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        return mod.add_kernel, mod.softmax_kernel

    @staticmethod
    def _run_case(kernel, kernel_name, signature, runtime_args, cfgs):
        methods = TestCostmodelBackendIntegration._load_pytorch_costmodel_methods()
        class DummyCfg:
            def __init__(self, kwargs, num_stages=2, num_warps=8):
                self.kwargs = kwargs
                self.num_stages = num_stages
                self.num_warps = num_warps
        class DummyAutotuner:
            pass
        dummy = DummyAutotuner()
        dummy.fn = kernel
        dummy.triton_meta = {"signature": signature, "constants": {}, "configs": None}
        dummy.inductor_meta = {"assert_indirect_indexing": True, "is_hip": False, "kernel_name": kernel_name}
        dummy.device_props = types.SimpleNamespace(type="npu", cc="")
        dummy.configs = [DummyCfg(x) for x in cfgs]
        dummy._make_ttir_module_from_cfg = types.MethodType(methods._make_ttir_module_from_cfg, dummy)
        dummy._try_parse_int_like = types.MethodType(methods._try_parse_int_like, dummy)
        dummy._build_ttir_arg_value_map = types.MethodType(methods._build_ttir_arg_value_map, dummy)
        dummy._build_costmodel_arg_bindings = types.MethodType(methods._build_costmodel_arg_bindings, dummy)
        dummy._build_costmodel_items = types.MethodType(methods._build_costmodel_items, dummy)
        items = dummy._build_costmodel_items(runtime_args, {})
        out = TestCostmodelBackendIntegration._costmodel_runtime.costmodel_bench(items)
        return items, out

    def test_inductor_costmodel_case_01_vector_add(self):
        add_kernel, _ = self._build_embedded_kernels()
        items, out = self._run_case(add_kernel, 'add_kernel', {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'output_ptr': '*fp32', 'n_elements': 'i32'}, [object(), object(), object(), 98432], [{'BLOCK_SIZE': 256}, {'BLOCK_SIZE': 1024}, {'BLOCK_SIZE': 2048}])
        self.assertTrue(items)
        self.assertIsInstance(out, dict)
        self.assertTrue(out)
        self.assertIn('arg3=', items[0].get('arg_bindings', ''))
        self.assertIn('pid_x=0', items[0].get('arg_bindings', ''))
        self.assertTrue(any(v != float('inf') for v in out.values()))

    def test_inductor_costmodel_case_02_fused_softmax(self):
        _, softmax_kernel = self._build_embedded_kernels()
        sig = {'output_ptr': '*fp32', 'input_ptr': '*fp32', 'input_row_stride': 'i32', 'output_row_stride': 'i32', 'n_rows': 'i32', 'n_cols': 'i32'}
        items, out = self._run_case(softmax_kernel, 'softmax_kernel', sig, [object(), object(), 781, 781, 1823, 781], [{'BLOCK_SIZE': 512}, {'BLOCK_SIZE': 1024}, {'BLOCK_SIZE': 2048}])
        self.assertTrue(items)
        self.assertIsInstance(out, dict)
        self.assertTrue(out)
        bindings = items[0].get('arg_bindings', '')
        self.assertIn('arg2=781', bindings)
        self.assertIn('arg3=781', bindings)
        self.assertIn('arg4=1823', bindings)
        self.assertIn('arg5=781', bindings)
        self.assertIn('pid_x=0', bindings)
        self.assertIn('num_programs_x=', bindings)
        self.assertTrue(any(v != float('inf') for v in out.values()))


if __name__ == "__main__":
    unittest.main()
