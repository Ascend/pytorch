import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


class TestNPUFastLaunchStatic(unittest.TestCase):
    def test_codegen_integration_is_installed_only_by_patch(self):
        wrapper = (REPO_ROOT / "torch_npu/_inductor/codegen/wrapper.py").read_text(
            encoding="utf-8"
        )
        triton_runtime = (
            REPO_ROOT / "torch_npu/_inductor/runtime/triton_heuristics.py"
        ).read_text(encoding="utf-8")
        config = (REPO_ROOT / "torch_npu/_inductor/config.py").read_text(
            encoding="utf-8"
        )
        inductor_init = (REPO_ROOT / "torch_npu/_inductor/__init__.py").read_text(
            encoding="utf-8"
        )
        patch = (
            REPO_ROOT
            / "torch_npu/_inductor/experimental/python_wrapper_fast_launch/patch.py"
        ).read_text(encoding="utf-8")
        bind = (
            REPO_ROOT
            / "torch_npu/_inductor/experimental/python_wrapper_fast_launch/bind.py"
        ).read_text(encoding="utf-8")
        emitter = (
            REPO_ROOT
            / "torch_npu/_inductor/experimental/python_wrapper_fast_launch/wrapper_codegen.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("python_wrapper_fast_launch", wrapper)
        self.assertNotIn("_npu_fast_launch", triton_runtime)
        self.assertRegex(
            config,
            r'enable_fast_launch\s*=\s*_parse_bool_env\(\s*'
            r'"TORCHINDUCTOR_NPU_FAST_LAUNCH",\s*False,\s*\)',
        )
        self.assertIn("patch_fast_launch()", inductor_init)
        self.assertIn("npu_config.enable_fast_launch", patch)
        self.assertIn("npu_config.enable_fast_launch", bind)
        self.assertIn("NPUPythonWrapperCodeGen", patch)
        self.assertIn("TritonCompileResultNpu", patch)
        self.assertIn("bind_python_wrapper_kernel_fast", patch)
        self.assertIn("call_slot=", emitter)
        self.assertIn("[None]", emitter)

    def test_package_import_does_not_eagerly_load_runtime_bindings(self):
        package_init = (
            REPO_ROOT
            / "torch_npu/_inductor/experimental/python_wrapper_fast_launch/__init__.py"
        ).read_text(encoding="utf-8")
        eager_imports = package_init.split("def __getattr__", 1)[0]
        self.assertNotIn("from .bind import", eager_imports)
        self.assertIn("def __getattr__(name)", package_init)

    @unittest.skip("temporarily disabled due to known CI failure")
    def test_cpp_backend_keeps_opcommand_and_runtime_guards(self):
        source = (
            REPO_ROOT
            / "torch_npu/_inductor/experimental/python_wrapper_fast_launch/csrc/bindings.cpp"
        ).read_text(
            encoding="utf-8"
        )
        self.assertIn("struct FastLaunchPlan", source)
        self.assertIn("packedArgsTemplate", source)
        self.assertIn("FastLaunchStaticWithPlan", source)
        self.assertIn("aclrtLaunchKernelWithHostArgs", source)
        self.assertIn("ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE", source)
        self.assertNotIn("rtKernelLaunch(", source)
        self.assertNotIn("rtKernelLaunchWithFlagV2", source)
        self.assertIn("OpCommand::RunOpApiV2", source)
        self.assertNotIn("SetCustomHandler", source)
        self.assertIn("grid product exceeds uint16 max", source)
        self.assertIn("args and arg_kinds size mismatch", source)

    def test_no_unplanned_or_operator_fast_launch_is_added(self):
        source = (
            REPO_ROOT
            / "torch_npu/_inductor/experimental/python_wrapper_fast_launch/csrc/bindings.cpp"
        ).read_text(
            encoding="utf-8"
        )
        self.assertNotIn('"_npu_inductor_fast_launch"', source)
        self.assertNotIn("OperatorFastLaunch", source)

    @unittest.skip("temporarily disabled due to known CI failure")
    def test_fast_launch_plan_precomputes_packed_argument_layout(self):
        source = (
            REPO_ROOT
            / "torch_npu/_inductor/experimental/python_wrapper_fast_launch/csrc/bindings.cpp"
        ).read_text(
            encoding="utf-8"
        )
        self.assertIn("struct FastLaunchArgLayout", source)
        self.assertIn("BuildPackedLayout(*plan)", source)
        self.assertIn("plan.argLayouts[index]", source)
        self.assertIn("plan.gridOffsets[index]", source)
        self.assertIn(
            "plan->packedArgsTemplate.resize(plan->packedArgsSize, 0)", source
        )
        self.assertEqual(source.count("packed.args = plan.packedArgsTemplate"), 2)
        self.assertIn("if (!plan.isPureSimt)", source)
        self.assertIn("if (plan.targetSupportFfts)", source)
        self.assertIn("rtGetC2cCtrlAddr(&fftsAddress, &fftsLength)", source)
        self.assertIn(
            "plan.packedArgsSize = AlignOffset(offset, packedAlignment)", source
        )
        self.assertIn(
            'TORCH_CHECK(alignment != 0, "alignment must be non-zero")', source
        )
        self.assertNotIn("void AppendBytes(", source)


if __name__ == "__main__":
    unittest.main()
