# Copyright (c) 2026, Huawei Technologies Co., Ltd
# Owner(s): ["module: inductor"]

from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest import mock
import unittest

import torch
from torch import nn
from torch._inductor.utils import run_and_get_code

import torch_npu  # noqa: F401


class _Buffer:
    def __init__(self):
        self.lines = []

    def writeline(self, line):
        self.lines.append(line)

    @contextmanager
    def indent(self):
        yield

    def text(self):
        return "\n".join(self.lines)


class TestTritonCppWrapper(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    def test_backend_registration_includes_cpp_wrapper(self):
        from torch_npu._inductor.triton_experimental import device
        from torch_npu._inductor.triton_experimental.codegen.cpp_wrapper import (
            TritonCppWrapperNpu,
        )

        with (
            mock.patch.object(device, "register_backend_for_device") as register,
            mock.patch(
                "torch_npu._inductor.triton_experimental.codegen.triton.NPUTritonScheduling"
            ),
        ):
            device.register_backend_for_npu()

        self.assertEqual(register.call_args.args[0], "npu")
        self.assertIs(register.call_args.args[3], TritonCppWrapperNpu)

    def test_lazy_compile_reports_required_option(self):
        from torch._inductor import config
        from torch_npu._inductor.triton_experimental.codegen.cpp_wrapper import (
            TritonCppWrapperNpu,
        )

        with config.patch({"triton.autotune_at_compile_time": False}):
            with self.assertRaisesRegex(
                NotImplementedError, "autotune_at_compile_time=True"
            ):
                TritonCppWrapperNpu()

    def test_torchbench_enables_compile_time_autotune_for_cpp_wrapper(self):
        import importlib.util
        import os
        import sys
        import types
        from pathlib import Path
        from types import SimpleNamespace

        from torch._inductor import config

        torchbench_dir = (
            Path(__file__).resolve().parents[3] / "benchmarks" / "torchbench"
        )
        spec = importlib.util.spec_from_file_location(
            "torchbench_common_cpp_wrapper_test", torchbench_dir / "common.py"
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        common = importlib.util.module_from_spec(spec)

        npu_support = types.ModuleType("npu_support")
        npu_support.patch_model = mock.Mock()
        profiler = types.ModuleType("profiler")
        profiler.NPUProfiler = mock.Mock()
        profiler.CUDAProfiler = mock.Mock()
        with mock.patch.dict(
            sys.modules, {"npu_support": npu_support, "profiler": profiler}
        ):
            spec.loader.exec_module(common)

        def configure(npu_backend):
            args = SimpleNamespace(
                cpp_wrapper=True,
                npu_backend=npu_backend,
                disable_aclgraph=False,
                dynamic_shapes=False,
                dynamic_batch_only=False,
                aclgraph_mode=None,
                only=None,
                backend="inductor",
                nopython=True,
                mfusion=False,
            )
            with (
                mock.patch.dict(os.environ),
                config.patch(
                    {
                        "cpp_wrapper": False,
                        "triton.autotune_at_compile_time": False,
                    }
                ),
            ):
                common.configure_compile_options(args, runner=object())
                return (
                    config.cpp_wrapper,
                    config.triton.autotune_at_compile_time,
                )

        self.assertEqual(configure("triton_experimental"), (True, True))
        self.assertEqual(configure("dvm"), (True, False))

    def test_a5_dispatch_recipe_is_emitted_as_cpp(self):
        from torch_npu._inductor.triton_experimental.codegen.cpp_wrapper import (
            _emit_grid,
        )

        output = _Buffer()
        params = {
            "def_args": ["xnumel"],
            "runtime_blocks": {},
            "config": {"XBLOCK": 256},
            "triton_meta": {"signature": {"xnumel": "i64", "XBLOCK": "constexpr"}},
            "inductor_meta": {
                "npu_dispatch_recipe": {
                    "lines": ["tiles = (xnumel + XBLOCK - 1) // XBLOCK"],
                    "factors": ["tiles"],
                }
            },
        }
        with mock.patch(
            "torch_npu._inductor.triton_experimental.codegen.cpp_wrapper.device_props.is_a5",
            return_value=True,
        ):
            _emit_grid(output, params)

        self.assertIn("constexpr int64_t XBLOCK = 256;", output.text())
        self.assertIn("uint32_t grid_0", output.text())

        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("a host C++ compiler is required")
        source = (
            "#include <algorithm>\n#include <cstdint>\nint main() {\n"
            "int64_t xnumel = 257;\n"
            + output.text()
            + "\nreturn grid_0 == 2 && grid_1 == 1 && grid_2 == 1 ? 0 : 1;\n}"
        )
        with tempfile.TemporaryDirectory() as root:
            cpp, binary = Path(root) / "grid.cpp", Path(root) / "grid"
            cpp.write_text(source)
            subprocess.run(
                [compiler, "-std=c++17", str(cpp), "-o", str(binary)],
                check=True,
                capture_output=True,
                timeout=60,
            )
            subprocess.run([str(binary)], check=True, timeout=10)

    def test_grid_policy_matches_shared_npu_policy(self):
        from torch_npu._inductor.triton_experimental.codegen.cpp_wrapper import (
            _emit_grid,
        )

        cases = [
            (
                "te_combo",
                {"te_combo_meta": {}, "npu_dispatch_recipe": {}},
                ["xnumel"],
                {},
                {"XBLOCK": 128},
                True,
                "uint32_t grid_0 = static_cast<uint32_t>(48);",
            ),
            (
                "linearized_scalar_reduction",
                {"npu_num_x_nodes": 0},
                ["xnumel"],
                {},
                {"XBLOCK": 128},
                False,
                "uint32_t grid_0 = static_cast<uint32_t>(1);",
            ),
            (
                "nonlinearized_reduction",
                {"npu_num_x_nodes": 0, "npu_linearize": False},
                ["xnumel"],
                {},
                {"XBLOCK": 128},
                False,
                "static_cast<uint32_t>(std::max<int64_t>(1, (xnumel + 128 - 1) / 128))",
            ),
            (
                "clamped_pointwise",
                {"npu_num_x_nodes": 1},
                ["xnumel"],
                {},
                {"XBLOCK": 128},
                False,
                "std::min<int64_t>((xnumel + 128 - 1) / 128, 48)",
            ),
        ]
        for name, metadata, def_args, runtime_blocks, config, is_a5, expected in cases:
            with self.subTest(name=name):
                output = _Buffer()
                signature_names = set(def_args) | set(runtime_blocks)
                if "reduction" in name:
                    signature_names.add("R0_BLOCK")
                params = {
                    "def_args": def_args,
                    "runtime_blocks": runtime_blocks,
                    "config": config,
                    "inductor_meta": metadata,
                    "triton_meta": {"signature": dict.fromkeys(signature_names, "i64")},
                }
                with (
                    mock.patch(
                        "torch_npu._inductor.triton_experimental.codegen.cpp_wrapper.device_props.is_a5",
                        return_value=is_a5,
                    ),
                    mock.patch(
                        "torch_npu._inductor.triton_experimental.codegen.cpp_wrapper.device_props.get_npu_vector_core_count",
                        return_value=48,
                    ),
                ):
                    _emit_grid(output, params)
                self.assertIn(expected, output.text())

    def test_cpp_builder_accepts_current_upstream_keyword(self):
        from torch._inductor.cpp_builder import get_cpp_torch_device_options

        options = get_cpp_torch_device_options("cpu", cpp_stdlib="libstdc++")
        self.assertEqual(len(options), 7)

    def test_binary_save_reuses_default_npu_cache_format(self):
        from torch_npu._inductor.runtime.triton_heuristics import NPUCachingAutotuner
        from torch_npu._inductor.triton_experimental.npu_triton_heuristics import (
            NPUCachingAutotuner as ExperimentalAutotuner,
        )

        autotuner = object.__new__(ExperimentalAutotuner)
        launcher = object()
        with mock.patch.object(
            NPUCachingAutotuner, "save_npu_kernel"
        ) as save_npu_kernel:
            autotuner.save_gpu_kernel(7, launcher)

        save_npu_kernel.assert_called_once_with(autotuner, 7, launcher)

    def test_binary_save_updates_benchmark_kernel_to_winner(self):
        from torch_npu._inductor.triton_experimental.npu_triton_heuristics import (
            NPUCachingAutotuner,
        )

        autotuner = object.__new__(NPUCachingAutotuner)
        candidate = mock.Mock(store_cubin=True)
        winner = mock.Mock(store_cubin=True)
        autotuner.launchers = [candidate]
        autotuner.inductor_meta = {"use_fast_triton_launcher": False}
        autotuner.cuda_kernel_saved = False

        def save(stream, launcher):
            autotuner.cuda_kernel_saved = True

        autotuner.save_gpu_kernel = mock.Mock(side_effect=save)
        autotuner.run(1, stream=7, benchmark_run=True)
        autotuner.run(1, stream=7, benchmark_run=True)
        autotuner.launchers = [winner]
        autotuner.run(1, stream=7)
        autotuner.run(1, stream=7)
        self.assertEqual(
            autotuner.save_gpu_kernel.call_args_list,
            [mock.call(7, candidate), mock.call(7, winner)],
        )
        winner.assert_called_with(1, stream=7)

    def test_cpp_wrapper_downcast_writeback(self):
        if not torch.npu.is_available():
            self.skipTest("NPU is required")

        def mutate(x):
            x.add_(3)
            return x * 2

        base = torch.arange(48, device="npu", dtype=torch.int64).view(6, 8)
        expected_base = base.clone()
        options = {"npu_backend": "triton_experimental"}
        python_fn = torch.compile(mutate, options=options)
        cpp_fn = torch.compile(mutate, options={**options, "cpp_wrapper": True})
        expected = python_fn(expected_base[1::2, 1::2])
        actual, codes = run_and_get_code(cpp_fn, base[1::2, 1::2])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(base, expected_base, rtol=0, atol=0)
        self.assertTrue(any("_downcast = te_downcast(" in code for code in codes))

    def test_cpp_wrapper_run_and_any_dims_shim(self):
        if not torch.npu.is_available():
            self.skipTest("NPU is required")

        class TinyAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = nn.Linear(32, 96)
                self.out = nn.Linear(32, 32)

            def forward(self, x):
                q, k, v = self.qkv(x).chunk(3, dim=-1)
                scores = torch.matmul(q, k.transpose(-2, -1)) * (32**-0.5)
                probs = torch.softmax(scores, dim=-1)
                return self.out(torch.matmul(probs, v)) + x

        model = TinyAttention().to("npu").eval()
        x = torch.randn(2, 8, 32, device="npu")
        expected = model(x)
        compiled = torch.compile(
            model,
            options={
                "npu_backend": "triton_experimental",
                "cpp_wrapper": True,
                "triton.autotune_at_compile_time": True,
            },
        )
        actual, codes = run_and_get_code(compiled, x)

        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
        self.assertTrue(any("cpp_wrapper_src" in code for code in codes))
        cpp_code = next(code for code in codes if "cpp_wrapper_src" in code)
        self.assertIn("aoti_torch_npu_any_dims", cpp_code)


if __name__ == "__main__":
    unittest.main()
