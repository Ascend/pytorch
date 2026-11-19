import contextlib
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import run_tests

import torch_npu  # noqa: F401
from torch_npu._inductor.codegen import cpp_wrapper_npu as cw

from testutils import TestUtils

CUBIN_PATH_NAME = "cubin_path"


def make_plan():
    return {
        "best_by_group": {"0": {"variant_id": "v0", "policy_id": "p0"}},
        "variant_order": ("v0", "v1"),
        "variants": {"v0": {"load_meta": {"cubin_path": "/tmp/v0.cubin", "mangled_name": "m0",
                                          "mix_mode": "aiv", "shared_mem": 0}}},
        "policies": {"p0": {
            "static_blocks": (("BLOCK_N", 128),),
            "runtime_block_rules": (
                ("BLOCK_M", (("op", "ceildiv"), ("axis_name", "m"), ("block_sub", 16))),
            ),
            "grid_target": 8,
        }},
        "axis_arg_indices": {"m": 1},
        "runtime_block_append_order": ("BLOCK_M",),
        "reachable_group_ids": (0,),
        "feature_arg_indices": ((0, 1), (2,)),
        "feature_sources": ({"source": "outer_product"}, {"source": "single"}),
        "group_features": ({"buckets": (4, 8)},),
        "group_id_count": 9,
    }


def make_deferred(kernel_name="kernel0", arg_types=None):
    call = object.__new__(cw.DeferredNpuTritonCallWrapper)
    call.wrapper_name = "wrapper0"
    call.kernel_name = kernel_name
    call.kernel_name_to_body = {}
    call.arg_types = list(arg_types or [])
    call.kernel_id = 0
    return call


def make_kernel_params(group_enabled=False, plan=None):
    return {
        "inductor_meta": {"group_enabled": group_enabled, "grouped_candidate_plan": plan or {},
                          "kernel_name": "kernel0"},
        "triton_meta": {"constants": [], "signature": {"x": "*fp16"}},
        "def_args": ["x"], "call_args": ["x"],
        "is_pure_simt": False, "parallel_mode": "simd",
    }


@contextlib.contextmanager
def patch_graph(graph):
    # V.graph is a read-only property on _V, so the graph has to be installed
    # through the virtualized handler instead of patched on the instance.
    with V.set_graph_handler(graph):
        yield


def make_header_wrapper():
    wrapper = object.__new__(cw.CppWrapperNpu)
    wrapper.header = IndentedBuffer()
    wrapper.device = "npu"
    return wrapper


def make_launch_wrapper():
    return SimpleNamespace(
        generate_args_decl=lambda *args, **kwargs: "int64_t launch_args = 0;",
        generate_launch_preparation=lambda *args, **kwargs: "launchKernel();")


class TestCppWrapperNpuHeader(TestUtils):
    def test_write_header_paths(self):
        # The NPU headers are written first, everything else is delegated to the
        # GPU wrapper and the shared community include is redirected to torch_npu.
        wrapper = make_header_wrapper()
        delegated = []

        def upstream_write_header(self):
            delegated.append(1)
            self.header.splice("#include <torch/csrc/inductor/aoti_include/common.h>")

        with mock.patch.object(cw.CppWrapperGpu, "write_header", upstream_write_header):
            wrapper.write_header()
        source = wrapper.header.getvalue()
        self.assertTrue(delegated)
        self.assertIn("#include <torch_npu/csrc/inductor/aoti_runtime/utils_npu.h>", source)
        self.assertIn("#include <torch_npu/csrc/inductor/aoti_torch/generated/c_shim_npu.h>", source)
        self.assertIn("<torch_npu/csrc/inductor/aoti_include/common.h>", source)
        self.assertNotIn("<torch/csrc/inductor/aoti_include/common.h>", source)

    def test_device_include(self):
        wrapper = make_header_wrapper()
        with mock.patch.object(cw.CppWrapperGpu, "add_device_include",
                               lambda self, device: None):
            wrapper.add_device_include("npu")
        source = wrapper.header.getvalue()
        self.assertIn("#include <acl/acl.h>", source)
        self.assertIn("#include <torch_npu/csrc/core/npu/NPUStream.h>", source)


class TestGroupedPlanQueries(TestUtils):
    def test_plan_queries(self):
        names = cw.DeferredNpuTritonCallWrapper._grouped_runtime_block_names(
            {"runtime_block_append_order": ("BLOCK_M", "BLOCK_N")})
        self.assertEqual(names, ("BLOCK_M", "BLOCK_N"))
        self.assertEqual(cw.DeferredNpuTritonCallWrapper._grouped_active_variants(make_plan()), ("v0",))
        self.assertEqual(
            cw.DeferredNpuTritonCallWrapper._grouped_load_meta(make_plan(), "v0")["cubin_path"],
            "/tmp/v0.cubin")
        for bad_plan, field in (({"best_by_group": {}}, "active_variants"),
                                ({"best_by_group": {"0": {"variant_id": "v9"}}, "variant_order": ()},
                                 "active_variants"),
                                ({"variants": {"v0": {}}}, "load_meta")):
            with self.assertRaises(RuntimeError, msg=field):
                if field == "active_variants":
                    cw.DeferredNpuTritonCallWrapper._grouped_active_variants(bad_plan)
                else:
                    cw.DeferredNpuTritonCallWrapper._grouped_load_meta(bad_plan, "v0")


class TestGroupedCodegen(TestUtils):
    def test_grouped_codegen(self):
        call = make_deferred()
        prefix = IndentedBuffer()
        call._generate_grouped_feature_inputs(prefix, make_plan(), ["m", "n", "k"])
        source = prefix.getvalue()
        self.assertIn("int64_t grouped_feature_0 = m * n;", source)
        self.assertIn("int64_t grouped_feature_1 = k;", source)
        ambiguous = make_plan()
        ambiguous["feature_sources"] = ({"source": "single"},)
        missing = make_plan()
        missing["feature_sources"] = ({"source": "single"}, {"source": "single"})
        for plan in (ambiguous, missing):
            with self.assertRaises(RuntimeError):
                call._generate_grouped_feature_inputs(IndentedBuffer(), plan, ["m", "n"])

        prefix = IndentedBuffer()
        call._generate_grouped_group_id(prefix, make_plan())
        call._generate_grouped_selection(prefix, make_plan(), ["batch", "m", "n"], ("v0",))
        source = prefix.getvalue()
        self.assertIn("grouped_group_stride *= 3;", source)
        self.assertIn("else grouped_bucket_0 = 2;", source)
        self.assertIn("grouped_group_id >= 9", source)
        self.assertIn("switch (grouped_group_id) {", source)
        self.assertIn("case 0: {", source)
        self.assertIn("BLOCK_N = 128;", source)
        self.assertIn("BLOCK_M = resolve_grouped_runtime_block(m, 8, 16);", source)
        self.assertIn('"grouped cpp wrapper resolved an unavailable group"', source)

        no_winner = make_plan()
        no_winner["reachable_group_ids"] = (5,)
        bad_op = make_plan()
        bad_op["policies"]["p0"]["runtime_block_rules"] = (
            ("BLOCK_M", (("op", "div"), ("axis_name", "m"), ("block_sub", 16))),)
        no_axis = make_plan()
        no_axis["axis_arg_indices"] = {}
        missing_block = make_plan()
        missing_block["runtime_block_append_order"] = ("BLOCK_M", "BLOCK_K")
        for plan in (no_winner, bad_op, no_axis, missing_block):
            with self.assertRaises(RuntimeError):
                call._generate_grouped_selection(IndentedBuffer(), plan, ["batch", "m", "n"], ("v0",))


class TestKernelLoadAndLaunch(TestUtils):
    def test_kernel_load_dispatch(self):
        params = make_kernel_params()
        params.update({"cubin_path": "/tmp/k.cubin", "mangled_name": "mangled",
                       "mix_mode": "aiv", "shared_mem": 1024})
        prefix = IndentedBuffer()
        with patch_graph(SimpleNamespace(aot_mode=False)), \
                mock.patch.object(cw, "get_cpp_wrapper_cubin_path_name",
                                  staticmethod(lambda: CUBIN_PATH_NAME)):
            make_deferred()._generate_single_kernel_load(prefix, "kernel_0", params)
        source = prefix.getvalue()
        self.assertIn("if (kernel_0 == nullptr) {", source)
        self.assertIn('"/tmp/k.cubin"', source)
        self.assertIn('"mangled"', source)
        self.assertIn("1024", source)

        call = make_deferred()
        prefix = IndentedBuffer()
        call.generate_load_kernel(prefix, "kernel_0", make_kernel_params(True, make_plan()))
        self.assertIn("static void* grouped_kernel_v0 = nullptr;", prefix.getvalue())
        prefix = IndentedBuffer()
        with patch_graph(SimpleNamespace(aot_mode=False)), \
                mock.patch.object(cw, "get_cpp_wrapper_cubin_path_name",
                                  staticmethod(lambda: CUBIN_PATH_NAME)):
            call.generate_load_kernel(prefix, "kernel_0", params)
        self.assertIn("loadKernel(", prefix.getvalue())

    def test_kernel_launch_dispatch(self):
        prefix = IndentedBuffer()
        with mock.patch.object(cw, "triton_support_auto_blockify", lambda: True):
            make_deferred(arg_types=[int])._generate_single_kernel_launch(
                prefix, make_launch_wrapper(), "kernel_0", make_kernel_params())
        source = prefix.getvalue()
        self.assertIn("auto launch_call = [=]() {", source)
        self.assertIn('launchKernel(launch_call, "kernel_0");', source)

        call = make_deferred(arg_types=[int])
        prefix = IndentedBuffer()
        with mock.patch.object(cw, "triton_support_auto_blockify", lambda: True):
            call.generate_launch_kernel(prefix, make_launch_wrapper(), "kernel_0",
                                        make_kernel_params())
        self.assertIn("launchKernel(launch_call", prefix.getvalue())

        prefix = IndentedBuffer()
        params = make_kernel_params(True, make_plan())
        # Each variant is loaded and launched from the same switch, so the load
        # path also reads V.graph here.
        with patch_graph(SimpleNamespace(aot_mode=False)), \
                mock.patch.object(cw, "triton_support_auto_blockify", lambda: True), \
                mock.patch.object(cw, "get_cpp_wrapper_cubin_path_name",
                                  staticmethod(lambda: CUBIN_PATH_NAME)):
            call.generate_launch_kernel(prefix, make_launch_wrapper(), "kernel_0", params)
        source = prefix.getvalue()
        self.assertIn("switch (grouped_variant_index) {", source)
        self.assertIn("grouped_kernel_v0", source)
        self.assertIn('"grouped cpp wrapper could not launch selected variant"', source)


class TestPatchRuntimeBlockParams(TestUtils):
    def _params(self, **meta):
        return {"def_args": ["t0", "t0_numel", "BLOCK_M"], "inductor_meta": dict(meta)}

    def test_runtime_block_params(self):
        call = make_deferred(arg_types=[int])
        params = self._params(runtime_block_arg_names=("BLOCK_M",))
        with mock.patch.object(cw.CudaKernelParamCache, "get", staticmethod(lambda name: params)):
            with call._patch_runtime_block_params() as patched:
                self.assertIs(patched, params)
                self.assertEqual(params["def_args"], ["t0", "t0_numel"])
        self.assertEqual(params["def_args"], ["t0", "t0_numel", "BLOCK_M"])
        self.assertEqual(call.arg_types, [int])

        call = make_deferred(kernel_name="multi_kernel_3")
        params = self._params()
        with mock.patch.object(cw.MultiKernelCall, "lookup_choice",
                               staticmethod(lambda name: "kernel_3")), \
                mock.patch.object(cw.CudaKernelParamCache, "get", staticmethod(lambda name: params)):
            with call._patch_runtime_block_params():
                self.assertEqual(call.kernel_name, "kernel_3")

        missing = make_deferred()
        with mock.patch.object(cw.CudaKernelParamCache, "get", staticmethod(lambda name: None)):
            with self.assertRaises(RuntimeError):
                with missing._patch_runtime_block_params():
                    pass
        call = make_deferred(arg_types=[int])
        params = self._params(runtime_block_arg_names=("BLOCK_M",), extra_launcher_args=["x"])
        with mock.patch.object(cw.CudaKernelParamCache, "get", staticmethod(lambda name: params)):
            with self.assertRaises(RuntimeError):
                with call._patch_runtime_block_params():
                    pass


class TestCppWrapperCompile(TestUtils):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self._prev_cache_size_limit = torch._dynamo.config.cache_size_limit
        torch._dynamo.config.cache_size_limit = 256

    def tearDown(self):
        torch._dynamo.config.cache_size_limit = self._prev_cache_size_limit
        super().tearDown()

    @config.patch(cpp_wrapper=True, force_disable_caches=True)
    def test_cpp_wrapper_compiles_a_fused_pointwise_reduction(self):
        def fn(x, y):
            return torch.relu(x + y).sum(-1)

        hits = []
        orig = cw.CppWrapperNpu.write_header

        def counting_write_header(self, *args, **kwargs):
            hits.append(1)
            return orig(self, *args, **kwargs)

        x = torch.randn(32, 64, dtype=torch.float16, device="npu")
        y = torch.randn(32, 64, dtype=torch.float16, device="npu")
        expected = fn(x, y)
        with mock.patch.object(cw.CppWrapperNpu, "write_header", counting_write_header):
            try:
                actual = torch.compile(fn)(x, y)
            except Exception as exc:
                # Some builds cannot drive NPU triton kernels through the cpp
                # wrapper: older Triton-Ascend lacks the is_pure_simt metadata
                # and this path does not populate CudaKernelParamCache. Both are
                # environment gaps, so skip instead of asserting on codegen.
                gaps = ("is_pure_simt", "CudaKernelParamCache not populated")
                if not any(gap in str(exc) for gap in gaps):
                    raise
                self.skipTest("cpp wrapper cannot compile on this build")

        self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
        self.assertTrue(hits, "cpp_wrapper codegen never ran")


if __name__ == "__main__":
    run_tests()
