# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Owner(s): ["module: inductor"]
"""Unit and in-process acceptance tests for the NPU static launcher.

Follow test_triton_experimental_enable.py / test_codecache.py locally and
upstream test_static_triton_launcher.py::TestStaticTritonCompileResult /
test_codecache.py::test_cache_load_function.

Missing static support fails the positive tests. FXGraph cache tests reset
in-memory state and reload the disk cache; they do NOT prove fresh-process or
worker recovery. Raw-only recovery is covered separately at the unit layer.
Startup environment-variable parsing is outside this in-process suite.
User-defined @triton.autotune currently routes through upstream heuristics, so
it is not claimed as experimental-backend coverage. Candidate selection and
release are covered through NPUCachingAutotuner.run with controlled timings.
"""

import gc
import inspect
import os
import pickle
import tempfile
import weakref
from contextlib import contextmanager, ExitStack, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._inductor import config
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.codecache import PyCodeCache
from torch._inductor.runtime.hints import HeuristicType
from torch._inductor.utils import clear_caches
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from triton.compiler.compiler import CompiledKernel
from triton.runtime import driver

from torch_npu._inductor import config as npu_config
from torch_npu._inductor.triton_experimental.npu_triton_heuristics import (
    NPUCachingAutotuner,
)
from torch_npu._inductor.triton_experimental import npu_triton_heuristics
from torch_npu._inductor.triton_experimental.static_launcher import adapter as static_adapter
from torch_npu._inductor.triton_experimental.static_launcher import compile_result as static_compile_result
from torch_npu._inductor.triton_experimental.static_launcher.adapter import (
    _arg_kind,
    NPUStaticArtifactAdapter,
    NPUStaticArtifactError,
)
from torch_npu._inductor.triton_experimental.static_launcher import (
    CannotStaticallyLaunchNPUKernel,
    NPUHostArgsLayout,
    NPULaunchMetadata,
    NPUStaticallyLaunchedTritonKernel,
    NPUStaticTritonCompileResult,
)
from torch_npu.testing.common_utils import SupportedDevices


class _LoadedKernel:
    """Only the external CANN owner is faked; Python ownership stays real."""

    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.closed = False

    def close(self):
        if not self.closed:
            self.events.append(("unload", self.name))
            self.closed = True

    def __del__(self):
        self.close()


class _LauncherBinding:
    def __init__(self):
        self.events = []

    def _load_kernel(self, binary, name, *metadata):
        self.events.append(("load", name))
        return _LoadedKernel(name, self.events)

    def _unload_kernel(self, owner):
        owner.close()

    def _launch_kernel(self, owner, grid0, grid1, grid2, stream, args):
        if owner.closed:
            raise AssertionError("launch after unload")
        self.events.append(("launch", owner.name, (grid0, grid1, grid2), stream, args))


def _generate_npu_tensor(shape, dtype):
    tensor_dtype = getattr(torch, dtype)
    if dtype in ("float32", "float16", "bfloat16"):
        return torch.randn(shape, dtype=tensor_dtype, device="npu") * 2000
    if dtype in ("int32", "int64"):
        return torch.randint(0, 2000, shape, dtype=tensor_dtype, device="npu")
    if dtype == "bool":
        return torch.randint(0, 2, shape, device="npu").bool()
    raise ValueError(f"Unsupported NPU tensor dtype: {dtype}")


@instantiate_parametrized_tests
class TestTritonExperimentalStaticLauncherUnit(TestCase):
    def setUp(self):
        super().setUp()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(config.patch({"cpp_wrapper": False}))
        # Eligibility unit tests do not depend on the installed CANN/library.
        stack.enter_context(mock.patch.object(static_compile_result, "_C", SimpleNamespace(
            _StaticNpuLauncher=SimpleNamespace(_is_supported=lambda: True),
        )))
        stack.enter_context(mock.patch.dict(os.environ, {
            "TRITON_REGISTER_TENSOR_MSPROF": "false",
            "TRITON_DEVICE_PRINT": "false",
            "TRITON_FAST_PTR": "false",
        }))

    @classmethod
    def _make_result(cls, binding, name="test_kernel", num_warps=4):
        kernel = cls._kernel(
            name=name, kernel_hash=name,
            launcher_arg_names=(), runtime_arg_names=(),
        )
        # The mocked loader reads raw bytes; this only bypasses disk recovery.
        kernel.npubin_path = "unused.npubin"
        kernel._c_impl = binding
        kernel.num_warps = num_warps
        return NPUStaticTritonCompileResult(
            kernel, triton.Config({}, num_warps=num_warps),
            {"device": 0, "constants": {}, "signature": {}}, {},
        )

    @staticmethod
    def _make_launcher(result):
        knobs = SimpleNamespace(runtime=SimpleNamespace(
            launch_enter_hook=None, launch_exit_hook=None,
        ))
        with (
            mock.patch.object(static_compile_result, "get_npu_vector_core_count", return_value=1),
            mock.patch.object(npu_triton_heuristics, "knobs", knobs),
        ):
            return result.make_launcher()

    @staticmethod
    def _metadata(**overrides):
        values = {
            "schema_version": 1,
            "host_args_layout": NPUHostArgsLayout.TRITON_ASCEND_3_2,
            "target_arch": "Ascend910B",
            "mix_mode": "aiv",
            "parallel_mode": "simd",
            "enable_simt": False,
            "shared_mem_dynamic_size": 0,
            "is_pure_simt": False,
            "target_support_ffts": False,
            "num_ctas": 1,
            "cluster_dims": (1, 1, 1),
            "workspace_size": 0,
            "lock_num": 0,
            "device_print_enabled": False,
        }
        values.update(overrides)
        return NPULaunchMetadata(**values)

    @classmethod
    def _kernel(
        cls,
        *,
        name="test_kernel",
        kernel_hash="hash",
        npubin_raw=b"npubin",
        launcher_arg_names=("out", "XBLOCK", "xnumel"),
        runtime_arg_names=("out", "xnumel"),
    ):
        return NPUStaticallyLaunchedTritonKernel(
            name=name,
            npubin_raw=npubin_raw,
            npubin_path=None,
            kernel_hash=kernel_hash,
            arg_names=launcher_arg_names,
            launcher_arg_names=launcher_arg_names,
            runtime_arg_names=runtime_arg_names,
            declared_constexprs=(1,) if launcher_arg_names else (),
            full_constexprs=(1,) if launcher_arg_names else (),
            arg_kinds=tuple(
                "tensor" if arg_name == "out" else "i32"
                for arg_name in runtime_arg_names
            ),
            launch_metadata=cls._metadata(),
            device=0,
            num_warps=4,
            shared=0,
            n_regs=0,
            n_spills=0,
        )

    @staticmethod
    def _binary(*, arg_names=(), signature=None, constexprs=(), constants=None):
        params = [
            SimpleNamespace(num=index, is_constexpr=index in constexprs)
            for index in range(len(arg_names))
        ]
        fn = SimpleNamespace(arg_names=arg_names, params=params)
        return SimpleNamespace(
            asm={"npubin": b"npubin"},
            src=SimpleNamespace(
                fn=fn,
                signature={} if signature is None else signature,
                constants={} if constants is None else constants,
            ),
            metadata=SimpleNamespace(
                target=SimpleNamespace(arch="Ascend910B"),
                name="test_kernel",
                mix_mode="aiv",
                parallel_mode="simd",
                force_simt_only=False,
                is_pure_simt=False,
                shared_mem_dynamic_size=0,
                num_warps=4,
            ),
            metadata_group={},
            name="test_kernel",
            hash="hash",
            shared=0,
            n_regs=0,
            n_spills=0,
        )

    @staticmethod
    def _autotuner_with_static_npubin(npubin_raw):
        autotuner = object.__new__(NPUCachingAutotuner)
        result = object.__new__(NPUStaticTritonCompileResult)
        result.kernel = SimpleNamespace(npubin_raw=npubin_raw)
        autotuner.compile_results = [result]
        return autotuner, result

    @config.patch("keep_static_cubin_raw", False)
    def test_prepare_for_caching_drops_npubin_raw_by_default(self):
        autotuner, result = self._autotuner_with_static_npubin(b"npubin")
        autotuner.prepare_for_caching()
        self.assertIsNone(result.kernel.npubin_raw)

    @config.patch("keep_static_cubin_raw", True)
    def test_prepare_for_caching_keeps_npubin_raw_when_configured(self):
        autotuner, result = self._autotuner_with_static_npubin(b"npubin")
        autotuner.prepare_for_caching()
        self.assertEqual(result.kernel.npubin_raw, b"npubin")

    @config.patch(
        {
            "use_static_triton_launcher": True,
            "strict_static_triton_launcher": True,
        }
    )
    @parametrize("hook_name", ["launch_enter_hook", "launch_exit_hook"])
    def test_legacy_triton_launch_hook_rejects_static_launcher(self, hook_name):
        class LegacyBinary:
            launch_enter_hook = None
            launch_exit_hook = None

        setattr(LegacyBinary, hook_name, object())
        with (
            mock.patch("torch._inductor.runtime.triton_compat.knobs", None),
            self.assertRaisesRegex(
                CannotStaticallyLaunchNPUKernel, hook_name.replace("_", " ") + " enabled"
            ),
        ):
            NPUStaticTritonCompileResult.can_statically_launch(
                LegacyBinary(),
                compile_meta={},
                inductor_meta={},
                heuristic_type=HeuristicType.POINTWISE,
            )

    @config.patch(
        {
            "use_static_triton_launcher": True,
            "strict_static_triton_launcher": True,
        }
    )
    @parametrize("hook_name", ["launch_enter_hook", "launch_exit_hook"])
    def test_modern_triton_launch_hook_rejects_static_launcher(self, hook_name):
        knobs = SimpleNamespace(
            runtime=SimpleNamespace(
                launch_enter_hook=None,
                launch_exit_hook=None,
            )
        )
        setattr(knobs.runtime, hook_name, SimpleNamespace(calls=[object()]))
        with (
            mock.patch("torch._inductor.runtime.triton_compat.knobs", knobs),
            self.assertRaisesRegex(
                CannotStaticallyLaunchNPUKernel, hook_name.replace("_", " ") + " enabled"
            ),
        ):
            NPUStaticTritonCompileResult.can_statically_launch(
                object(),
                compile_meta={},
                inductor_meta={},
                heuristic_type=HeuristicType.POINTWISE,
            )

    @config.patch({"use_static_triton_launcher": True, "strict_static_triton_launcher": True})
    @parametrize("hook_api", ["legacy", "modern_none", "modern_empty"])
    def test_empty_launch_hooks_allow_static_launcher(self, hook_api):
        class Binary:
            launch_enter_hook = None
            launch_exit_hook = None

        hook = SimpleNamespace(calls=[]) if hook_api == "modern_empty" else None
        knobs = None if hook_api == "legacy" else SimpleNamespace(
            runtime=SimpleNamespace(launch_enter_hook=hook, launch_exit_hook=hook)
        )
        static_kernel = SimpleNamespace(npu_launch_metadata=self._metadata())
        with (
            mock.patch("torch._inductor.runtime.triton_compat.knobs", knobs),
            mock.patch.object(NPUStaticArtifactAdapter, "from_compiled_kernel", return_value=static_kernel),
        ):
            self.assertIs(
                NPUStaticTritonCompileResult.can_statically_launch(
                    Binary(), compile_meta={}, inductor_meta={},
                    heuristic_type=HeuristicType.POINTWISE,
                ),
                static_kernel,
            )

    @config.patch("use_static_triton_launcher", True)
    @parametrize("support", ["missing_api", "missing_binding", "old_binding"])
    @parametrize("strict", [False, True])
    def test_missing_runtime_support_strict_and_fallback(self, support, strict):
        binding = SimpleNamespace()
        if support == "missing_api":
            binding._is_supported = lambda: False
        c_module = SimpleNamespace()
        if support != "missing_binding":
            c_module._StaticNpuLauncher = binding
        with (
            config.patch("strict_static_triton_launcher", strict),
            mock.patch.object(static_compile_result, "_C", c_module),
            mock.patch.object(NPUStaticArtifactAdapter, "from_compiled_kernel") as adapt,
        ):
            if strict:
                with self.assertRaisesRegex(CannotStaticallyLaunchNPUKernel, "aclrtLaunchKernelWithHostArgs"):
                    NPUStaticTritonCompileResult.can_statically_launch(
                        object(), compile_meta={}, inductor_meta={},
                        heuristic_type=HeuristicType.POINTWISE,
                    )
            else:
                self.assertIsNone(NPUStaticTritonCompileResult.can_statically_launch(
                    object(), compile_meta={}, inductor_meta={},
                    heuristic_type=HeuristicType.POINTWISE,
                ))
            adapt.assert_not_called()

    @config.patch("use_static_triton_launcher", False)
    def test_disabled_static_launcher_does_not_probe_runtime(self):
        probe = mock.Mock(side_effect=AssertionError("unexpected runtime probe"))
        with mock.patch.object(static_compile_result, "_C", SimpleNamespace(
            _StaticNpuLauncher=SimpleNamespace(_is_supported=probe),
        )):
            self.assertIsNone(NPUStaticTritonCompileResult.can_statically_launch(
                object(), compile_meta={}, inductor_meta={},
                heuristic_type=HeuristicType.POINTWISE,
            ))
        probe.assert_not_called()

    @config.patch("use_static_triton_launcher", True)
    @parametrize("name_source", ["inductor_meta", "binary", "missing"])
    def test_ineligible_kernel_strict_and_fallback(self, name_source):
        binary = SimpleNamespace(name="compiled_kernel") if name_source != "missing" else object()
        inductor_meta = {"kernel_name": "triton_test_kernel"} if name_source == "inductor_meta" else {}
        expected_name = {
            "inductor_meta": "triton_test_kernel",
            "binary": "compiled_kernel",
            "missing": "unknown",
        }[name_source]
        static_kernel = SimpleNamespace(
            npu_launch_metadata=self._metadata(workspace_size=1)
        )
        knobs = SimpleNamespace(
            runtime=SimpleNamespace(
                launch_enter_hook=None,
                launch_exit_hook=None,
            )
        )
        for strict in (False, True):
            with (
                self.subTest(strict=strict),
                config.patch("strict_static_triton_launcher", strict),
                mock.patch("torch._inductor.runtime.triton_compat.knobs", knobs),
                mock.patch.object(
                    NPUStaticArtifactAdapter,
                    "from_compiled_kernel",
                    return_value=static_kernel,
                ),
                self.assertLogs("torch._inductor", level="INFO") as captured,
            ):
                if strict:
                    with self.assertRaisesRegex(
                        CannotStaticallyLaunchNPUKernel, "workspace is required"
                    ):
                        NPUStaticTritonCompileResult.can_statically_launch(
                            binary,
                            compile_meta={},
                            inductor_meta=inductor_meta,
                            heuristic_type=HeuristicType.POINTWISE,
                        )
                else:
                    self.assertIsNone(
                        NPUStaticTritonCompileResult.can_statically_launch(
                            binary,
                            compile_meta={},
                            inductor_meta=inductor_meta,
                            heuristic_type=HeuristicType.POINTWISE,
                        )
                    )
            self.assertIn(
                f"Bypassing NPU static Triton launcher for kernel {expected_name} "
                "due to workspace is required",
                "\n".join(captured.output),
            )

    @config.patch(
        {
            "use_static_triton_launcher": True,
            "strict_static_triton_launcher": True,
            "static_launch_user_defined_triton_kernels": False,
        }
    )
    def test_user_defined_kernel_policy(self):
        with self.assertRaisesRegex(
            CannotStaticallyLaunchNPUKernel, "user-defined Triton kernel"
        ):
            NPUStaticTritonCompileResult.can_statically_launch(
                object(),
                compile_meta={},
                inductor_meta={},
                heuristic_type=HeuristicType.USER_AUTOTUNE,
            )

        static_kernel = SimpleNamespace(npu_launch_metadata=self._metadata())
        knobs = SimpleNamespace(
            runtime=SimpleNamespace(
                launch_enter_hook=None,
                launch_exit_hook=None,
            )
        )
        with (
            config.patch("static_launch_user_defined_triton_kernels", True),
            mock.patch("torch._inductor.runtime.triton_compat.knobs", knobs),
            mock.patch.object(
                NPUStaticArtifactAdapter,
                "from_compiled_kernel",
                return_value=static_kernel,
            ),
        ):
            self.assertIs(
                NPUStaticTritonCompileResult.can_statically_launch(
                    object(),
                    compile_meta={},
                    inductor_meta={},
                    heuristic_type=HeuristicType.USER_AUTOTUNE,
                ),
                static_kernel,
            )

    def test_adapter_argument_kinds(self):
        expected = {
            "*fp32": "tensor",
            "i1": "bool",
            "i8": "i8",
            "i16": "i16",
            "i32": "i32",
            "i64": "i64",
            "u1": "u32",
            "u8": "u8",
            "u16": "u16",
            "u32": "u32",
            "u64": "u64",
            "fp16": "f32",
            "bf16": "f32",
            "fp32": "f32",
            "f32": "f32",
            "fp64": "f64",
        }
        for signature, kind in expected.items():
            with self.subTest(signature=signature):
                self.assertEqual(_arg_kind(signature), kind)
        with self.assertRaisesRegex(
            NPUStaticArtifactError, "Tensor descriptor arguments are unsupported"
        ):
            _arg_kind("tensordesc<2>")

    def test_adapter_supports_no_runtime_arguments(self):
        with mock.patch(
            "torch_npu._inductor.triton_experimental.static_launcher.adapter."
            "_target_supports_ffts",
            return_value=False,
        ):
            kernel = NPUStaticArtifactAdapter.from_compiled_kernel(
                self._binary(), {}, {}
            )
        self.assertEqual(kernel.arg_names, ())
        self.assertEqual(kernel.launcher_arg_names, ())
        self.assertEqual(kernel.runtime_arg_names, ())
        self.assertEqual(kernel.arg_kinds, ())

    @parametrize("modern_triton", [False, True])
    @parametrize("signature_key", ["index", "name", "tuple"])
    def test_adapter_filters_declared_and_implied_constants(self, modern_triton, signature_key):
        binary = self._binary(
            arg_names=("out", "XBLOCK", "xnumel", "implied"),
            signature={0: "*fp32", 1: "constexpr", 2: "i32", 3: "i32"},
            constexprs=(1,),
            constants={1: 128, 3: 1},
        )
        if signature_key != "index":
            def convert_key(index):
                return binary.src.fn.arg_names[index] if signature_key == "name" else (index,)

            binary.src.signature = {convert_key(key): value for key, value in binary.src.signature.items()}
            binary.src.constants = {convert_key(key): value for key, value in binary.src.constants.items()}
        with (
            mock.patch.object(static_adapter, "_target_supports_ffts", return_value=False),
            mock.patch.object(static_adapter, "IS_TRITON_36_PLUS", modern_triton),
        ):
            kernel = NPUStaticArtifactAdapter.from_compiled_kernel(
                binary,
                {"constants": {"XBLOCK": 128}},
                {},
            )
        self.assertEqual(kernel.runtime_arg_names, ("out", "xnumel"))
        self.assertEqual(kernel.arg_kinds, ("tensor", "i32"))
        self.assertEqual(kernel.launcher_arg_names, ("out", "xnumel", "implied"))
        self.assertEqual(kernel.runtime_arg_indices, (0, 1))

    def test_adapter_maps_simt_and_ffts_metadata(self):
        binary = self._binary()
        binary.metadata.parallel_mode = "simt"
        binary.metadata.is_pure_simt = True
        binary.metadata.shared_mem_dynamic_size = 4096
        with (
            mock.patch.object(static_adapter, "IS_TRITON_36_PLUS", True),
            mock.patch(
                "torch_npu._inductor.triton_experimental.static_launcher.adapter."
                "_target_supports_ffts",
                return_value=True,
            ),
        ):
            kernel = NPUStaticArtifactAdapter.from_compiled_kernel(binary, {}, {})
        metadata = kernel.npu_launch_metadata
        self.assertTrue(metadata.enable_simt)
        self.assertTrue(metadata.is_pure_simt)
        self.assertTrue(metadata.target_support_ffts)
        self.assertEqual(metadata.shared_mem_dynamic_size, 4096)
        self.assertEqual(
            metadata.host_args_layout,
            NPUHostArgsLayout.TRITON_ASCEND_3_6,
        )
        self.assertEqual(metadata.trailing_pointer_count, 3)

    @parametrize(
        "scratch_size_field",
        ["global_scratch_size", "profile_scratch_size"],
    )
    @parametrize("scratch_size", [0, 4096])
    def test_adapter_allows_zero_and_rejects_nonzero_scratch_requirement(
        self, scratch_size_field, scratch_size
    ):
        binary = self._binary()
        setattr(binary.metadata, scratch_size_field, scratch_size)
        contexts = (
            self.assertRaisesRegex(
                NPUStaticArtifactError,
                f"{scratch_size_field} is required but unsupported",
            )
            if scratch_size
            else nullcontext()
        )
        with (
            mock.patch.object(static_adapter, "IS_TRITON_36_PLUS", True),
            mock.patch.object(
                static_adapter, "_target_supports_ffts", return_value=False
            ),
            contexts,
        ):
            NPUStaticArtifactAdapter.from_compiled_kernel(binary, {}, {})

    @parametrize("modern_triton", [False, True])
    @parametrize("pure_simt", [False, True])
    def test_adapter_selects_host_args_layout(self, modern_triton, pure_simt):
        binary = self._binary()
        binary.metadata.parallel_mode = "simt" if pure_simt else "simd"
        binary.metadata.force_simt_only = pure_simt
        binary.metadata.is_pure_simt = pure_simt
        with (
            mock.patch.object(static_adapter, "IS_TRITON_36_PLUS", modern_triton),
            mock.patch.object(
                static_adapter, "_target_supports_ffts", return_value=False
            ),
        ):
            kernel = NPUStaticArtifactAdapter.from_compiled_kernel(binary, {}, {})

        metadata = kernel.npu_launch_metadata
        expected_layout = (
            NPUHostArgsLayout.TRITON_ASCEND_3_6
            if modern_triton
            else NPUHostArgsLayout.TRITON_ASCEND_3_2
        )
        expected_trailing_pointers = (
            3 if modern_triton and pure_simt else int(modern_triton)
        )
        self.assertEqual(metadata.host_args_layout, expected_layout)
        self.assertEqual(metadata.is_pure_simt, pure_simt)
        self.assertEqual(metadata.trailing_pointer_count, expected_trailing_pointers)

    @parametrize("modern_triton", [False, True])
    @parametrize(
        "field",
        ["mix_mode", "parallel_mode", "shared_mem_dynamic_size", "pure_simt"],
    )
    def test_adapter_rejects_missing_required_metadata(self, modern_triton, field):
        binary = self._binary()
        metadata_field = field
        if field == "pure_simt":
            metadata_field = "is_pure_simt" if modern_triton else "force_simt_only"
        delattr(binary.metadata, metadata_field)
        with (
            mock.patch.object(static_adapter, "IS_TRITON_36_PLUS", modern_triton),
            mock.patch.object(
                static_adapter, "_target_supports_ffts", return_value=False
            ),
            self.assertRaisesRegex(NPUStaticArtifactError, metadata_field),
        ):
            NPUStaticArtifactAdapter.from_compiled_kernel(binary, {}, {})

    def test_launch_metadata_validation(self):
        invalid = (
            ("schema_version", 2, "metadata schema"),
            ("host_args_layout", "unknown", "host args layout"),
            ("target_arch", "Ascend910A", "910B-or-newer"),
            ("mix_mode", "mix", "mix mode"),
            ("parallel_mode", "", "parallel_mode is missing"),
            ("shared_mem_dynamic_size", -1, "non-negative"),
            ("workspace_size", -1, "non-negative"),
            ("lock_num", -1, "non-negative"),
            ("is_pure_simt", True, "requires enable_simt"),
            ("num_ctas", 2, "num_ctas=1"),
            ("cluster_dims", (2, 1, 1), "cluster_dims"),
        )
        metadata = self._metadata()
        metadata.validate()
        for target_arch in (
            "Ascend910B",
            "Ascend910D",
            "Ascend910_93",
            "Ascend910_95",
            "Ascend950",
        ):
            with self.subTest(target_arch=target_arch):
                replace(metadata, target_arch=target_arch).validate()
        for field, value, error in invalid:
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, error):
                replace(metadata, **{field: value}).validate()

    def test_reload_npubin_from_raw_and_missing_artifact(self):
        with tempfile.TemporaryDirectory() as cache_dir, mock.patch(
            "torch_npu._inductor.triton_experimental.static_launcher.compile_result."
            "triton_cache_dir",
            return_value=cache_dir,
        ):
            kernel = self._kernel()
            result = object.__new__(NPUStaticTritonCompileResult)
            result.kernel = kernel
            result.compile_meta = {"device": 0}
            result.reload_npubin_path()
            self.assertEqual(Path(kernel.npubin_path).read_bytes(), b"npubin")

            missing_kernel = self._kernel(
                name="missing", kernel_hash="missing", npubin_raw=None
            )
            missing_result = object.__new__(NPUStaticTritonCompileResult)
            missing_result.kernel = missing_kernel
            missing_result.compile_meta = {"device": 0}
            with self.assertRaisesRegex(
                RuntimeError, "saved by TritonBundler not found"
            ):
                missing_result.reload_npubin_path()

    def test_static_kernel_pickle_round_trip(self):
        kernel = self._kernel()
        kernel.npubin_path = "/tmp/stale.npubin"
        # Lambdas cannot be pickled: accidentally retaining a live handle must fail.
        kernel.loaded_kernel = lambda: None
        kernel.function = lambda: None
        kernel._c_impl = lambda: None

        restored = pickle.loads(pickle.dumps(kernel))

        self.assertEqual(restored.runtime_arg_indices, (0, 2))
        self.assertEqual(restored.npubin_raw, b"npubin")
        self.assertIsNone(restored.npubin_path)
        self.assertIsNone(restored.loaded_kernel)
        self.assertIsNone(restored.function)
        self.assertIsNone(restored._c_impl)
        self.assertEqual(
            restored.npu_launch_metadata.host_args_layout,
            NPUHostArgsLayout.TRITON_ASCEND_3_2,
        )

    @parametrize(
        "layout,pure_simt,expected_trailing_pointers",
        [
            (NPUHostArgsLayout.TRITON_ASCEND_3_2, False, 0),
            (NPUHostArgsLayout.TRITON_ASCEND_3_2, True, 0),
            (NPUHostArgsLayout.TRITON_ASCEND_3_6, False, 1),
            (NPUHostArgsLayout.TRITON_ASCEND_3_6, True, 3),
        ],
    )
    def test_load_kernel_forwards_normalized_host_args_layout(
        self, layout, pure_simt, expected_trailing_pointers
    ):
        kernel = self._kernel()
        kernel.npu_launch_metadata = self._metadata(
            host_args_layout=layout,
            parallel_mode="simt" if pure_simt else "simd",
            enable_simt=pure_simt,
            is_pure_simt=pure_simt,
        )
        binding = mock.Mock()
        binding._load_kernel.return_value = object()
        kernel._c_impl = binding

        kernel.load_kernel()

        self.assertEqual(
            binding._load_kernel.call_args.args[-1], expected_trailing_pointers
        )

    def test_static_kernel_restores_legacy_pickle_state(self):
        state = self._kernel().__getstate__()
        del state["runtime_arg_indices"]
        del state["_c_impl"]
        restored = object.__new__(NPUStaticallyLaunchedTritonKernel)
        restored.__setstate__(state)
        self.assertEqual(restored.runtime_arg_indices, (0, 2))
        self.assertIsNone(restored._c_impl)
        self.assertEqual(restored.npubin_raw, b"npubin")

    def test_autotune_releases_loser_and_keeps_winner(self):
        binding = _LauncherBinding()
        winner_result = self._make_result(binding, "winner")
        loser_result = self._make_result(binding, "loser", num_warps=8)
        winner = self._make_launcher(winner_result)
        loser = self._make_launcher(loser_result)
        autotuner = object.__new__(NPUCachingAutotuner)
        # The winner is not the first config, so skipping selection cannot pass.
        autotuner.launchers = [loser, winner]
        autotuner.compile_results = [loser_result, winner_result]
        autotuner.fn = SimpleNamespace(__name__="test_autotune", arg_names=())
        autotuner.triton_meta = {"signature": {}}
        autotuner.device_props = SimpleNamespace(type="npu", index=0)
        autotuner.heuristic_type = HeuristicType.POINTWISE
        autotuner.size_hints = None
        autotuner.inductor_meta = {}
        autotuner.precompile_time_taken_ns = 0
        autotuner.save_cache_hook = None

        def benchmark():
            # Timings are controlled, but candidates use real generated launchers.
            self.assertEqual(len(autotuner.launchers), 2)
            for launcher in autotuner.launchers:
                launcher(stream=123)
            binding.events.append(("benchmark_complete",))
            return {loser: 2.0, winner: 1.0}

        with (
            mock.patch.object(autotuner, "benchmark_all_configs", side_effect=benchmark),
            mock.patch("torch._inductor.runtime.triton_heuristics.TritonBundler.put_winner") as put_winner,
        ):
            autotuner.run(stream=123)
        put_winner.assert_called_once_with(winner.cache_hash)

        self.assertEqual(autotuner.launchers, [winner])
        self.assertEqual(autotuner.compile_results, [winner_result])
        self.assertIsNone(loser_result.kernel.loaded_kernel)
        self.assertIsNotNone(winner_result.kernel.loaded_kernel)
        self.assertEqual(binding.events.count(("unload", "loser")), 1)
        self.assertNotIn(("unload", "winner"), binding.events)
        self.assertLess(
            binding.events.index(("benchmark_complete",)),
            binding.events.index(("unload", "loser")),
        )
        autotuner.run(stream=456)
        self.assertEqual(binding.events[-1], ("launch", "winner", (1, 1, 1), 456, ()))
        self.assertEqual(binding.events.count(("load", "winner")), 1)
        self.assertEqual(binding.events.count(("load", "loser")), 1)
        loser_result.kernel.close()
        self.assertEqual(binding.events.count(("unload", "loser")), 1)
        winner_result.kernel.close()
        # This tests ordering after benchmark returns, not CANN synchronization.

    def test_launcher_keeps_kernel_owner_alive(self):
        binding = _LauncherBinding()
        result = self._make_result(binding)
        launcher = self._make_launcher(result)
        owner_ref = weakref.ref(result.kernel)
        loaded_ref = weakref.ref(result.kernel.loaded_kernel)
        del result
        gc.collect()
        self.assertIsNotNone(owner_ref())
        self.assertIsNotNone(loaded_ref())
        self.assertNotIn(("unload", "test_kernel"), binding.events)
        launcher(stream=123)

        del launcher
        gc.collect()
        self.assertIsNone(owner_ref())
        self.assertIsNone(loaded_ref())
        self.assertEqual(binding.events.count(("unload", "test_kernel")), 1)

    @parametrize("modern_triton", [False, True])
    def test_no_args_generated_launcher(self, modern_triton):
        binding = _LauncherBinding()
        result = self._make_result(binding)
        with mock.patch.object(npu_triton_heuristics, "IS_TRITON_36_PLUS", modern_triton):
            launcher = self._make_launcher(result)
        launcher(stream=123)
        self.assertTrue(launcher._is_static)
        self.assertEqual(binding.events[-1], ("launch", "test_kernel", (1, 1, 1), 123, ()))
        result.kernel.close()

    def test_close_unloads_kernel_once(self):
        unloaded = []

        class FakeImpl:
            @staticmethod
            def _unload_kernel(kernel):
                unloaded.append(kernel)

        kernel = self._kernel()
        loaded_kernel = object()
        kernel.loaded_kernel = loaded_kernel
        kernel.function = loaded_kernel
        kernel._c_impl = FakeImpl

        kernel.close()
        kernel.close()

        self.assertEqual(unloaded, [loaded_kernel])
        self.assertIsNone(kernel.loaded_kernel)
        self.assertIsNone(kernel.function)

    def test_run_accepts_only_runtime_arguments(self):
        calls = []

        class FakeImpl:
            @staticmethod
            def _launch_kernel(*args):
                calls.append(args)

        kernel = self._kernel()
        loaded_kernel = object()
        kernel.loaded_kernel = loaded_kernel
        kernel._c_impl = FakeImpl

        kernel.run(
            2,
            3,
            4,
            123,
            "out_ptr",
            17,
        )

        self.assertEqual(
            calls,
            [(loaded_kernel, 2, 3, 4, 123, ("out_ptr", 17))],
        )
        with self.assertRaisesRegex(RuntimeError, "unexpected argument count"):
            kernel.run(
                2,
                3,
                4,
                123,
                "out_ptr",
                128,
                17,
            )

    def _make_runtime_autotuner(self, launcher):
        def kernel_fn():
            pass

        kernel_fn.src = "def static_runtime_test(): pass"
        autotuner = NPUCachingAutotuner(
            fn=kernel_fn,
            triton_meta={
                "device": SimpleNamespace(
                    type="npu", index=0, warp_size=32, max_threads_per_block=1024,
                ),
                "signature": {}, "constants": {},
            },
            configs=[launcher.config], save_cache_hook=None,
            mutated_arg_names=[], optimize_mem=False,
            heuristic_type=HeuristicType.POINTWISE,
            inductor_meta={"use_fast_triton_launcher": False},
        )
        autotuner.launchers = [launcher]
        return autotuner

    def test_static_codegen_is_independent_and_uses_global_runner(self):
        binding = _LauncherBinding()
        result = self._make_result(binding)
        result.inductor_meta["extra_launcher_args"] = ["extra"]
        with mock.patch.object(
            npu_triton_heuristics.NPUTritonCompileResult, "make_launcher",
            side_effect=AssertionError("static must not call the ordinary generator"),
        ):
            launcher = self._make_launcher(result)
        self.assertEqual(list(inspect.signature(launcher).parameters), ["extra", "stream"])
        self.assertEqual(launcher._expected_positional_count, 1)
        self.assertIsNone(launcher.__defaults__)
        self.assertIsNone(launcher.__closure__)
        self.assertIs(launcher.__globals__["runner"].__self__, result.kernel)
        launcher(42, stream=123)
        self.assertEqual(binding.events[-1], ("launch", "test_kernel", (1, 1, 1), 123, ()))

    def test_static_codegen_filters_specialized_arguments(self):
        binding = _LauncherBinding()
        kernel = NPUStaticallyLaunchedTritonKernel(
            name="filtered", npubin_raw=b"npubin", npubin_path="unused.npubin",
            kernel_hash="filtered", arg_names=("out", "XBLOCK", "implied", "xnumel", "none"),
            launcher_arg_names=("out", "implied", "xnumel", "none"),
            runtime_arg_names=("out", "xnumel"), declared_constexprs=(1,),
            full_constexprs=(1, 2, 4), arg_kinds=("tensor", "i32"),
            launch_metadata=self._metadata(), device=0, num_warps=4,
            shared=0, n_regs=0, n_spills=0,
        )
        kernel._c_impl = binding
        result = NPUStaticTritonCompileResult(
            kernel, triton.Config({"XBLOCK": 128}, num_warps=4),
            {"device": 0, "constants": {"XBLOCK": 128, "implied": 7, "none": None},
             "signature": {"out": "*fp32", "xnumel": "i32"}}, {},
        )
        launcher = self._make_launcher(result)
        for count in (257, 513):
            launcher("output", 7, count, None, stream=123)
            self.assertEqual(binding.events[-1][-1], ("output", count))

    @parametrize("modern_triton", [False, True])
    @parametrize("has_launch_metadata", [False, True])
    def test_shared_codegen_preserves_ordinary_runner_abi(self, modern_triton, has_launch_metadata):
        class Binary:
            launch_enter_hook = None
            launch_exit_hook = None

        calls = []

        def runner(*args):
            calls.append(args)

        binary = Binary()
        binary.src = SimpleNamespace(fn=SimpleNamespace(
            arg_names=["out", "xnumel", "XBLOCK"], constexprs=[2],
        ))
        binary._init_handles = mock.Mock()
        binary.run = runner
        binary.function = object()
        binary.packed_metadata = {"test": "metadata"}
        binary.num_warps = 4
        binary.shared = 0
        binary.num_ctas = 1
        binary.cluster_dims = (1, 1, 1)
        if has_launch_metadata:
            binary.launch_metadata = mock.Mock(side_effect=AssertionError("no hooks"))
        result = npu_triton_heuristics.NPUTritonCompileResult(
            binary, triton.Config({"XBLOCK": 128}, num_warps=4),
            {"constants": {"XBLOCK": 128}, "signature": {"out": "*fp32", "xnumel": "i32"}},
            {"npu_num_x_nodes": 1},
        )
        knobs = SimpleNamespace(runtime=SimpleNamespace(
            launch_enter_hook=None, launch_exit_hook=None,
        ))
        with (
            mock.patch.object(npu_triton_heuristics, "IS_TRITON_36_PLUS", modern_triton),
            mock.patch.object(npu_triton_heuristics, "knobs", knobs),
            mock.patch.object(npu_triton_heuristics, "get_npu_vector_core_count", return_value=40),
        ):
            launcher = result.make_launcher()
        self.assertIn("runner", inspect.signature(launcher).parameters)
        self.assertNotIn("XBLOCK", inspect.signature(launcher).parameters)
        self.assertFalse(getattr(launcher, "_is_static", False))
        launcher("output", 257, stream=123)
        if has_launch_metadata:
            expected = (3, 1, 1, 123, binary.function, binary.packed_metadata,
                        None, None, None, "output", 257)
        else:
            expected = (3, 1, 1, 4, 1, 1, 1, 1, 0, 123, binary.function,
                        None, None, binary.packed_metadata, "output", 257)
        self.assertEqual(calls, [expected])
        binary._init_handles.assert_called_once_with()

    @parametrize("fast_enabled", [False, True])
    def test_static_run_preserves_npu_dispatch(self, fast_enabled):
        binding = _LauncherBinding()
        result = self._make_result(binding)
        autotuner = self._make_runtime_autotuner(self._make_launcher(result))
        autotuner.inductor_meta["use_fast_triton_launcher"] = fast_enabled
        with mock.patch.object(
            npu_triton_heuristics.CachingAutotuner, "run",
            side_effect=AssertionError("ABI alignment must not migrate the run lifecycle"),
        ):
            autotuner.run(stream=123)
            fast_run = autotuner.__dict__["run"]
            autotuner.run(stream=456)
            self.assertIs(autotuner.__dict__["run"], fast_run)
        self.assertEqual(
            [event for event in binding.events if event[0] == "launch"],
            [("launch", "test_kernel", (1, 1, 1), stream, ()) for stream in (123, 456)],
        )

    def test_static_benchmark_preserves_npu_dispatch(self):
        binding = _LauncherBinding()
        result = self._make_result(binding)
        autotuner = self._make_runtime_autotuner(self._make_launcher(result))
        with mock.patch.object(
            npu_triton_heuristics.CachingAutotuner, "run",
            side_effect=AssertionError("benchmark must use the NPU run lifecycle"),
        ):
            autotuner.run(stream=123)
            self.assertIn("run", autotuner.__dict__)
            autotuner.run(stream=456, benchmark_run=True)
            self.assertNotIn("run", autotuner.__dict__)
            autotuner.run(stream=789)
            self.assertIn("run", autotuner.__dict__)
        self.assertEqual(
            [event[3] for event in binding.events if event[0] == "launch"],
            [123, 456, 789],
        )

    def test_static_launcher_list_replacement_uses_new_winner(self):
        binding = _LauncherBinding()
        first = self._make_launcher(self._make_result(binding, "first"))
        second = self._make_launcher(self._make_result(binding, "second"))
        autotuner = self._make_runtime_autotuner(first)
        autotuner.run(stream=123)
        autotuner.launchers = [second]
        autotuner.run(stream=456)
        self.assertEqual(binding.events[-1], ("launch", "second", (1, 1, 1), 456, ()))

    def test_static_fast_run_preserves_downcast_writeback(self):
        def body(out):
            self.assertEqual(out.dtype, torch.int32)
            out.fill_(9)

        scope = {"body": body}
        exec("def launcher(out, stream):\n    body(out)\n", scope)
        launcher = scope["launcher"]
        launcher._is_static = True
        launcher._npu_def_args = ["out"]
        launcher._expected_positional_count = 1
        launcher.config = triton.Config({})
        wrapped = npu_triton_heuristics._wrap_launcher_with_downcast(
            launcher, {"out": "*i64"}, {"out"},
        )
        autotuner = self._make_runtime_autotuner(wrapped)
        self.assertEqual(wrapped._expected_positional_count, 1)
        out = torch.zeros(4, dtype=torch.int64)
        for value in (1, 2):
            out.fill_(value)
            autotuner.run(out, stream=123)
            self.assertTrue(torch.equal(out, torch.full_like(out, 9)))
            self.assertIn("run", autotuner.__dict__)


@instantiate_parametrized_tests
class TestTritonExperimentalStaticLauncherIntegration(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.npu.is_available():
            self.skipTest("requires an available NPU")

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache_root = Path(temporary.name)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.dict(os.environ, {
            "TORCHINDUCTOR_CACHE_DIR": str(self.cache_root / "inductor"),
            "TRITON_CACHE_DIR": str(self.cache_root / "triton"),
            "TRITON_INTERPRET": "0",
            "TRITON_DEVICE_PRINT": "false",
            "TRITON_REGISTER_TENSOR_MSPROF": "false",
            "TRITON_FAST_PTR": "false",
        }))
        stack.enter_context(config.patch({
            "compile_threads": 1,
            "cpp_wrapper": False,
            "triton.store_cubin": False,
            "use_static_triton_launcher": True,
            "strict_static_triton_launcher": True,
            "keep_static_cubin_raw": False,
            "fx_graph_cache": False,
            "fx_graph_remote_cache": False,
            "bundle_triton_into_fx_graph_cache": False,
            "autotune_remote_cache": False,
            "autotune_local_cache": True,
            "coordinate_descent_tuning": False,
            "force_disable_caches": False,
        }))
        stack.enter_context(functorch_config.patch({
            "enable_autograd_cache": False,
            "enable_remote_autograd_cache": False,
        }))
        # Reusing the low-level plan must not require the old FastLaunch switch.
        stack.enter_context(mock.patch.object(npu_config, "enable_fast_launch", False))
        self.addCleanup(counters.clear)
        self.addCleanup(self._reset_memory_cache)
        self._reset_memory_cache()
        counters.clear()

    @staticmethod
    def _reset_memory_cache():
        torch._dynamo.reset()
        # Keep disk entries for the cache-hit tests. Do not use purge=True.
        PyCodeCache.cache_clear()
        clear_caches()  # Includes CompiledTritonKernels.
        gc.collect()

    @contextmanager
    def _observe_launch(self, *, static=True, restore=False):
        original_init = CompiledKernel._init_handles
        original_triton = AsyncCompile.triton
        experimental_kernels = []

        def initialize(kernel, *args, **kwargs):
            if static:
                raise AssertionError("static path must not call _init_handles")
            return original_init(kernel, *args, **kwargs)

        def compile_triton(compiler, kernel_name, source_code, *args, **kwargs):
            if "npu_triton_heuristics" in source_code:
                experimental_kernels.append(kernel_name)
            return original_triton(compiler, kernel_name, source_code, *args, **kwargs)

        with (
            mock.patch.object(triton, "compile", wraps=triton.compile) as compile_mock,
            mock.patch.object(
                CompiledKernel, "_init_handles", autospec=True, side_effect=initialize,
            ) as init_mock,
            mock.patch.object(
                driver.active, "launcher_cls", wraps=driver.active.launcher_cls,
            ) as launcher_mock,
            mock.patch.object(AsyncCompile, "triton", autospec=True, side_effect=compile_triton),
        ):
            if static:
                launcher_mock.side_effect = AssertionError(
                    "static path must not construct the ordinary launcher"
                )
            if restore:
                compile_mock.side_effect = AssertionError(
                    "complete static cache hit must not call triton.compile"
                )
            yield

        # Positive evidence that compilation did not only select external ops.
        self.assertTrue(experimental_kernels)
        if static:
            init_mock.assert_not_called()
            launcher_mock.assert_not_called()
        else:
            self.assertGreater(init_mock.call_count, 0)
            self.assertGreater(launcher_mock.call_count, 0)
        if restore:
            compile_mock.assert_not_called()
        else:
            self.assertGreater(compile_mock.call_count, 0)

    def _check(self, fn, inputs, *, dynamic=False, static=True, restore=False):
        with self._observe_launch(static=static, restore=restore):
            compiled = torch.compile(
                fn, fullgraph=True, dynamic=dynamic,
                options={"npu_backend": "triton_experimental"},
            )
            for args in inputs:
                expected = fn(*args)
                actual = compiled(*args)
                torch.npu.synchronize()
                tolerance = 1e-3 if actual.is_floating_point() else 0
                # Correctness is already synchronized above. Compare on CPU so
                # the assertion framework does not enqueue unrelated NPU
                # comparison kernels (isclose/mask/sum).
                self.assertEqual(
                    actual.cpu(), expected.cpu(), atol=tolerance, rtol=tolerance
                )

    @staticmethod
    def _pointwise(x, y):
        return torch.relu(x + y)

    def _inputs(self, shape=(1024, 32), dtype="float32"):
        return tuple(_generate_npu_tensor(shape, dtype) for _ in range(2))

    def _load_static_kernel(self, compiled_kernel):
        device_interface = get_interface_for_device("npu")
        device = device_interface.current_device()
        kernel = NPUStaticArtifactAdapter.from_compiled_kernel(
            compiled_kernel,
            {"device": device},
            {},
        )
        kernel.load_kernel(device)
        self.addCleanup(kernel.close)
        return kernel

    @staticmethod
    def _run_static_kernel(kernel, *args, grid=(1, 1, 1)):
        device_interface = get_interface_for_device("npu")
        stream = device_interface.get_raw_stream(device_interface.current_device())
        kernel.run(
            *grid,
            stream,
            *args,
        )

    @parametrize("dtype", ["float32", "float16", "int32"])
    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_static_pointwise(self, dtype):
        self._check(self._pointwise, [self._inputs(dtype=dtype), self._inputs(dtype=dtype)])

    @SupportedDevices(["Ascend950"])
    def test_static_pure_simt_pointwise_abi(self):
        @triton.jit
        def pure_simt_pointwise(input_ptr, output_ptr, numel, BLOCK_SIZE: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel
            values = tl.load(input_ptr + offsets, mask=mask)
            tl.store(output_ptr + offsets, values * 2.0 + 1.0, mask=mask)

        numel = 1025
        block_size = 256
        grid = (triton.cdiv(numel, block_size), 1, 1)
        input_tensor = torch.randn(numel, dtype=torch.float32, device="npu")
        ordinary_output = torch.empty_like(input_tensor)
        compiled_kernel = pure_simt_pointwise[grid[:1]](
            input_tensor,
            ordinary_output,
            numel,
            BLOCK_SIZE=block_size,
            compile_mode="simt_only",
        )
        torch.npu.synchronize()

        expected = input_tensor * 2.0 + 1.0
        self.assertEqual(ordinary_output.cpu(), expected.cpu())

        kernel = self._load_static_kernel(compiled_kernel)
        self.assertTrue(kernel.npu_launch_metadata.is_pure_simt)
        self.assertTrue(kernel.npu_launch_metadata.enable_simt)
        static_output = torch.empty_like(input_tensor)
        self._run_static_kernel(
            kernel,
            input_tensor,
            static_output,
            numel,
            grid=grid,
        )
        torch.npu.synchronize()
        self.assertEqual(static_output.cpu(), expected.cpu())

    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_static_no_args_kernel(self):
        @triton.jit
        def kernel_no_op():
            pass

        compiled_kernel = kernel_no_op[(1,)]()
        kernel = self._load_static_kernel(compiled_kernel)
        self.assertEqual(kernel.arg_kinds, ())
        self._run_static_kernel(kernel)
        torch.npu.synchronize()

    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_static_signed_scalar_abi(self):
        @triton.jit(do_not_specialize=["arg1", "arg2", "arg3", "arg4"])
        def signed_scalars(
            output,
            arg1: tl.int8,
            arg2: tl.int16,
            arg3: tl.int32,
            arg4: tl.int64,
        ):
            tl.store(output, arg1.to(tl.int64))
            tl.store(output + 1, arg2.to(tl.int64))
            tl.store(output + 2, arg3.to(tl.int64))
            tl.store(output + 3, arg4)

        # Separate slots expose argument swaps, sign extension and i64 truncation.
        args = (-113, -30001, -(2**30) + 7, -(2**40) + 19)
        static_output = torch.zeros(4, dtype=torch.int64, device="npu")
        # Compile without executing the ordinary launcher: its signed i8
        # argument parser currently rejects negative values. Validate our ABI
        # against independent expected values instead.
        compiled_kernel = signed_scalars.warmup(static_output, *args, grid=(1,))

        kernel = self._load_static_kernel(compiled_kernel)
        self.assertEqual(kernel.arg_kinds, ("tensor", "i8", "i16", "i32", "i64"))
        self._run_static_kernel(kernel, static_output, *args)
        torch.npu.synchronize()
        self.assertEqual(static_output.cpu(), torch.tensor(args, dtype=torch.int64))

    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_static_unsigned_scalar_abi(self):
        @triton.jit(do_not_specialize=["arg1", "arg2", "arg3", "arg4"])
        def unsigned_scalars(
            output,
            arg1: tl.uint8,
            arg2: tl.uint16,
            arg3: tl.uint32,
            arg4: tl.uint64,
        ):
            tl.store(output, arg1.to(tl.int64))
            tl.store(output + 1, arg2.to(tl.int64))
            tl.store(output + 2, arg3.to(tl.int64))
            tl.store(output + 3, arg4.to(tl.int64))

        # Cross each signed-width boundary, including the Python -> u64 boundary.
        # Store u64's bit pattern as i64 to avoid needing NPU uint64 tensor ops.
        args = (241, 60013, 2**31 + 17, 2**63 + 29)
        expected = (*args[:3], args[3] - 2**64)
        ordinary_output = torch.zeros(4, dtype=torch.int64, device="npu")
        compiled_kernel = unsigned_scalars[(1,)](ordinary_output, *args)
        torch.npu.synchronize()
        self.assertEqual(
            ordinary_output.cpu(), torch.tensor(expected, dtype=torch.int64)
        )

        kernel = self._load_static_kernel(compiled_kernel)
        self.assertEqual(kernel.arg_kinds, ("tensor", "u8", "u16", "u32", "u64"))
        static_output = torch.zeros_like(ordinary_output)
        self._run_static_kernel(kernel, static_output, *args)
        torch.npu.synchronize()
        self.assertEqual(static_output.cpu(), ordinary_output.cpu())

    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_static_bool_and_float_scalar_abi(self):
        @triton.jit(do_not_specialize=["condition", "value"])
        def bool_float_scalars(output, condition: tl.int1, value: tl.float32):
            result = tl.where(condition, value, -value)
            tl.store(output, result)

        ordinary_output = torch.zeros(1, dtype=torch.float32, device="npu")
        compiled_kernel = bool_float_scalars[(1,)](ordinary_output, True, 3.25)
        torch.npu.synchronize()
        self.assertEqual(
            ordinary_output.cpu(), torch.tensor([3.25], dtype=torch.float32)
        )

        kernel = self._load_static_kernel(compiled_kernel)
        self.assertIn(
            kernel.arg_kinds,
            (
                ("tensor", "bool", "f32"),  # Triton 3.2: i1
                ("tensor", "u32", "f32"),  # Triton 3.6: u1
            ),
        )
        static_output = torch.zeros_like(ordinary_output)
        # Reuse one loaded kernel with both bool values and changed float inputs.
        for condition, value in ((True, 3.25), (False, 7.5), (True, -2.25)):
            self._run_static_kernel(kernel, static_output, condition, value)
            torch.npu.synchronize()
            self.assertEqual(
                static_output.cpu(),
                torch.tensor([value if condition else -value], dtype=torch.float32),
            )

    @parametrize(
        "signature,values,expected_kind",
        [
            ("u1", (False, True), "u32"),
            ("fp16", (1.5, -2.25), "f32"),
            ("bf16", (1.5, -2.25), "f32"),
        ],
    )
    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_static_narrow_scalar_abi(self, signature, values, expected_kind):
        if signature in ("fp16", "bf16"):
            self.skipTest("Triton Ascend does not yet support fp16/bf16 scalar ABI")

        @triton.jit
        def scalar_kernel(output, value):
            tl.store(output, value.to(tl.float32))

        source = triton.compiler.ASTSource(
            fn=scalar_kernel,
            signature={"output": "*fp32", "value": signature},
        )
        compiled_kernel = triton.compile(source)
        ordinary_runner = compiled_kernel[(1, 1, 1)]
        kernel = self._load_static_kernel(compiled_kernel)
        self.assertEqual(kernel.arg_kinds, ("tensor", expected_kind))

        ordinary_output = torch.zeros(1, dtype=torch.float32, device="npu")
        static_output = torch.zeros_like(ordinary_output)
        for value in values:
            ordinary_runner(ordinary_output, value)
            self._run_static_kernel(kernel, static_output, value)
            torch.npu.synchronize()
            self.assertEqual(ordinary_output.cpu(), torch.tensor([float(value)]))
            self.assertEqual(static_output.cpu(), ordinary_output.cpu())

    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_static_ineligible_falls_back_to_ordinary_launcher(self):
        # Force the eligibility failure without enabling GPU-specific save_cubin.
        with (
            config.patch("strict_static_triton_launcher", False),
            mock.patch.object(
                NPUStaticArtifactAdapter, "from_compiled_kernel",
                side_effect=NPUStaticArtifactError("test unsupported static artifact"),
            ) as adapt,
        ):
            self._check(self._pointwise, [self._inputs()], static=False)
        self.assertGreater(adapt.call_count, 0)

    @parametrize("strict", [False, True])
    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_disable_static_triton_launcher(self, strict):
        with config.patch({
            "use_static_triton_launcher": False,
            "strict_static_triton_launcher": strict,
        }):
            self._check(self._pointwise, [self._inputs()], static=False)

    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_static_toggle_in_same_process(self):
        with config.patch("use_static_triton_launcher", False):
            self._check(self._pointwise, [self._inputs()], static=False)
        self._reset_memory_cache()
        self._check(self._pointwise, [self._inputs()])

    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_static_broadcast_noncontiguous(self):
        batches = []
        for _ in range(2):
            x = torch.randn(65, 130, device="npu")[:, 1:].transpose(0, 1)
            y = torch.randn(1, 65, device="npu")
            self.assertFalse(x.is_contiguous())
            self.assertGreater(x.storage_offset(), 0)
            batches.append((x, y))
        self._check(self._pointwise, batches)

    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_static_reduction_tail(self):
        def fn(x, y):
            return (x + y).sum(dim=-1)

        batches = [
            (torch.randn(128, 257, device="npu"), torch.randn(128, 257, device="npu"))
            for _ in range(2)
        ]
        self._check(fn, batches)

    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_static_dynamic_shape(self):
        self._check(
            self._pointwise,
            [self._inputs(shape) for shape in [(1024, 32), (1537, 32), (1024, 32)]],
            dynamic=True,
        )

    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_static_empty_tensor(self):
        # Match upstream's mixed empty/nonempty case: a purely empty graph
        # could pass without ever exercising a Triton launcher.
        def fn(x, y):
            return torch.cat((x * 4, y + 10))

        self._check(fn, [(torch.rand(0, device="npu"), torch.rand(20, device="npu"))])

    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_static_nondefault_stream(self):
        with self._observe_launch():
            compiled = torch.compile(
                self._pointwise, fullgraph=True,
                options={"npu_backend": "triton_experimental"},
            )
            for _ in range(2):
                x, y = self._inputs()
                stream = torch.npu.Stream()
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    actual = compiled(x + 1, y)
                stream.synchronize()
                torch.npu.synchronize()
                expected = self._pointwise(x + 1, y)
                torch.npu.synchronize()
                self.assertEqual(actual.cpu(), expected.cpu())
        # Correctness smoke test, not a deterministic wrong-stream detector.

    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_static_npu_graph_capture_replay(self):
        with self._observe_launch():
            compiled = torch.compile(
                self._pointwise,
                fullgraph=True,
                options={"npu_backend": "triton_experimental"},
            )
            static_x, static_y = self._inputs()
            capture_stream = torch.npu.Stream()
            capture_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(capture_stream):
                compiled(static_x, static_y)
            capture_stream.synchronize()

            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=capture_stream):
                static_output = compiled(static_x, static_y)

            for _ in range(2):
                new_x, new_y = self._inputs()
                expected = self._pointwise(new_x, new_y)
                static_x.copy_(new_x)
                static_y.copy_(new_y)
                graph.replay()
                torch.npu.synchronize()
                self.assertEqual(
                    static_output.cpu(),
                    expected.cpu(),
                    atol=1e-3,
                    rtol=1e-3,
                )

    @parametrize("relocate_triton_cache", [False, True])
    @config.patch({"fx_graph_cache": True, "bundle_triton_into_fx_graph_cache": True})
    @SupportedDevices(["Ascend910B", "Ascend910_93", "Ascend950"])
    def test_static_fxgraph_cache_reload(self, relocate_triton_cache):
        # Follow upstream test_cache_load_function: cold run, reset memory
        # caches, warm run; check the community bundle counters as evidence.
        self._check(self._pointwise, [self._inputs()])
        self.assertGreater(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        self.assertGreater(counters["inductor"]["triton_bundler_save_static_autotuner"], 0)

        self._reset_memory_cache()
        counters.clear()
        with ExitStack() as stack:
            if relocate_triton_cache:
                old_cache = self.cache_root / "triton"
                self.assertTrue(old_cache.is_dir())
                old_cache.rename(self.cache_root / "triton-unavailable")
                stack.enter_context(mock.patch.dict(os.environ, {
                    "TRITON_CACHE_DIR": str(self.cache_root / "triton-new"),
                }))
            self._check(self._pointwise, [self._inputs()], restore=True)

        self.assertGreater(counters["inductor"]["fxgraph_cache_hit"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertGreater(counters["inductor"]["triton_bundler_load_static_autotuner"], 0)
        # keep_static_cubin_raw=False means this relocation is restored solely
        # from the bundled .npubin artifact. It does not claim cross-process or
        # raw-only fallback coverage.


if __name__ == "__main__":
    run_tests()
