import unittest
from types import SimpleNamespace

import torch
from torch._inductor import config
from torch._inductor.codecache import CudaKernelParamCache
from torch._inductor.utils import IndentedBuffer, run_and_get_cpp_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_npu._inductor.codegen.cpp_wrapper_npu import (
    CppWrapperNpu,
    DeferredNpuTritonCallWrapper,
)

GROUP_COUNT = 32
HIDDEN_SIZE = 1600
GATE_SIZE = 256
POST_SIZE = 64
SIDE_SIZE = 32
COMPILE_SEQUENCE_LENGTH = 200


def _variant_load_meta(variant_id):
    return {
        "cubin_path": f"/tmp/triton_kernel_{variant_id}.cubin",
        "mangled_name": f"mangled_{variant_id}",
        "mix_mode": "aiv",
        "shared_mem": 64,
        "parallel_mode": "vector",
        "force_simt_only": False,
        "shared_mem_dynamic_size": 0,
        "has_auto_blockify_blacklist_op": False,
    }


def _grouped_plan():
    return {
        "variants": {
            "v0": {"config": {}, "load_meta": _variant_load_meta("v0")},
            "v1": {"config": {}, "load_meta": _variant_load_meta("v1")},
        },
        "variant_order": ("v0", "v1"),
        "best_by_group": {
            "0": {"variant_id": "v0", "policy_id": "p0"},
            "1": {"variant_id": "v1", "policy_id": "p1"},
        },
        "policies": {
            "p0": {
                "grid_target": 8,
                "static_blocks": (),
                "runtime_block_rules": (
                    (
                        "XBLOCK",
                        (
                            ("op", "ceildiv"),
                            ("axis_name", "x"),
                            ("block_sub", 4),
                        ),
                    ),
                ),
            },
            "p1": {
                "grid_target": 16,
                "static_blocks": (),
                "runtime_block_rules": (
                    (
                        "XBLOCK",
                        (
                            ("op", "ceildiv"),
                            ("axis_name", "x"),
                            ("block_sub", 8),
                        ),
                    ),
                ),
            },
        },
        "runtime_block_append_order": ("XBLOCK",),
        "group_id_count": 2,
        "reachable_group_ids": (0, 1),
        "group_features": (
            {
                "name": "x",
                "source": "axis",
                "axis_names": ("x",),
                "buckets": (128,),
            },
        ),
        "axis_arg_indices": {"x": 1},
        "feature_arg_indices": ((1,),),
        "feature_sources": (
            {
                "name": "x",
                "source": "axis",
                "axis_names": ("x",),
            },
        ),
    }


def _render_grouped_wrapper(grouped_plan=None, *, return_files=False):
    grouped_plan = grouped_plan or _grouped_plan()
    inductor_meta = {
        "group_enabled": True,
        "grouped_candidate_plan": grouped_plan,
        "grid_type": "GridNpu",
        "kernel_name": "triton_kernel",
        "axis_names": ("x",),
        "runtime_block_arg_names": ("XBLOCK",),
        "primary_group_axis": "x",
    }
    params = {
        "def_args": ["in_ptr", "x_numel", "XBLOCK"],
        "call_args": ["in_ptr", "x_numel", "XBLOCK"],
        "config": {"split_axis": (0,), "split_blocks": (128,)},
        "inductor_meta": inductor_meta,
        "triton_meta": {
            "signature": {
                "in_ptr": "fp16",
                "x_numel": "i64",
                "XBLOCK": "i64",
            },
            "constants": {},
        },
        "mangled_name": "unused",
        "shared_mem": 64,
        "cubin_path": "/tmp/unused.cubin",
        "mix_mode": "aiv",
        "parallel_mode": "vector",
        "force_simt_only": False,
    }
    graph = SimpleNamespace(
        cpp_wrapper=True,
        aot_mode=False,
        is_const_graph=False,
        constant_reprs={},
        inputs_to_check=[],
        graph_input_names=[],
        graph_inputs={},
        device_types={"npu"},
        wrapper_code=None,
    )
    with V.set_graph_handler(graph):
        wrapper = CppWrapperNpu()
        graph.wrapper_code = wrapper
        wrapper.prefix = IndentedBuffer()
        CudaKernelParamCache.cache_clear()
        CudaKernelParamCache.cache["triton_kernel"] = params
        deferred = DeferredNpuTritonCallWrapper(
            wrapper_name="call_triton_kernel",
            kernel_name="triton_kernel",
            kernel_name_to_body={},
            arg_types=[torch.float16, int],
            tma_tensor_args={},
        )
        deferred.generate(wrapper)
        source = wrapper.prefix.getvalue()
        if return_files:
            return source, tuple(wrapper.additional_files)
        return source


class GatedTransposeBmmModel(torch.nn.Module):
    def forward(
        self,
        source,
        gate_down_rhs,
        gate_down_bias,
        gate_up_rhs,
        bias,
        post_bmm_rhs,
        side_bmm_rhs,
    ):
        transposed = source.transpose(0, 1)
        gate_down = torch.bmm(transposed, gate_down_rhs)
        gate_down = torch.tanh(gate_down + gate_down_bias)
        bmm_result = torch.bmm(gate_down, gate_up_rhs)
        gated = transposed * torch.tanh(bmm_result + bias)
        post = torch.bmm(gated, post_bmm_rhs)
        side = torch.bmm(transposed, side_bmm_rhs)
        return post, side


def _make_gated_transpose_inputs(sequence_length):
    def rand(*shape):
        return torch.randn(shape, device="npu:0", dtype=torch.float16)

    return (
        rand(sequence_length, GROUP_COUNT, HIDDEN_SIZE),
        rand(GROUP_COUNT, HIDDEN_SIZE, GATE_SIZE),
        rand(GROUP_COUNT, 1, GATE_SIZE),
        rand(GROUP_COUNT, GATE_SIZE, HIDDEN_SIZE),
        rand(GROUP_COUNT, 1, HIDDEN_SIZE),
        rand(GROUP_COUNT, HIDDEN_SIZE, POST_SIZE) * 0.01,
        rand(GROUP_COUNT, HIDDEN_SIZE, SIDE_SIZE),
    )


def _mark_sequence_length_dynamic(inputs):
    torch._dynamo.mark_dynamic(
        inputs[0],
        0,
        hint_override=COMPILE_SEQUENCE_LENGTH,
    )


class TestGroupedCppWrapper(TestCase):
    def tearDown(self):
        CudaKernelParamCache.cache_clear()
        torch._dynamo.reset()
        super().tearDown()

    def test_grouped_wrapper_emits_bucket_dispatch_and_variants(self):
        source = _render_grouped_wrapper()

        self.assertIn("switch (grouped_group_id)", source)
        self.assertIn("case 0: {", source)
        self.assertIn("case 1: {", source)
        self.assertIn("grouped_kernel_v0", source)
        self.assertIn("grouped_kernel_v1", source)
        self.assertIn('"mangled_v0"', source)
        self.assertIn('"mangled_v1"', source)

    def test_grouped_wrapper_materializes_block_sub_aligned_runtime_block(self):
        source = _render_grouped_wrapper()

        self.assertIn(
            "auto resolve_grouped_runtime_block",
            source,
        )
        self.assertIn(
            "total_subblocks = ceildiv(axis_numel, block_sub)",
            source,
        )
        self.assertIn(
            "program_subblocks = ceildiv(",
            source,
        )
        self.assertIn(
            "effective_grid = ceildiv(",
            source,
        )
        self.assertIn(
            "XBLOCK = resolve_grouped_runtime_block(x_numel, 8, 4)",
            source,
        )
        self.assertIn(
            "XBLOCK = resolve_grouped_runtime_block(x_numel, 16, 8)",
            source,
        )
        self.assertLess(source.index("int64_t XBLOCK"), source.index("uint32_t grid_0"))

    def test_grouped_wrapper_omits_unselected_variant(self):
        grouped_plan = _grouped_plan()
        grouped_plan["best_by_group"]["1"] = {
            "variant_id": "v0",
            "policy_id": "p0",
        }
        source, additional_files = _render_grouped_wrapper(
            grouped_plan,
            return_files=True,
        )

        self.assertIn("grouped_kernel_v0", source)
        self.assertNotIn("grouped_kernel_v1", source)
        self.assertIn("/tmp/triton_kernel_v0.cubin", additional_files)
        self.assertNotIn("/tmp/triton_kernel_v1.cubin", additional_files)
        self.assertNotIn("/tmp/unused.cubin", additional_files)

    @unittest.skipIf(not torch.npu.is_available(), "NPU is not available")
    def test_gated_transpose_dynamic_shapes_functionality_and_accuracy(self):
        import torch_npu._inductor.config as npu_config

        previous_group_autotune = (
            npu_config.enable_symbolic_shape_group_autotune
        )
        npu_config.enable_symbolic_shape_group_autotune = True
        try:
            with config.patch(
                {
                    "cpp_wrapper": True,
                    "compile_threads": 1,
                    "force_disable_caches": True,
                }
            ):
                model = GatedTransposeBmmModel().eval()
                compile_inputs = _make_gated_transpose_inputs(
                    COMPILE_SEQUENCE_LENGTH
                )
                _mark_sequence_length_dynamic(compile_inputs)
                compiled = torch.compile(
                    model,
                    backend="inductor",
                    fullgraph=True,
                    dynamic=None,
                )

                with torch.no_grad():
                    expected = model(*compile_inputs)
                    actual, cpp_code = run_and_get_cpp_code(
                        compiled, *compile_inputs
                    )
                    torch.npu.synchronize()
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=0.02,
                    atol=0.03,
                    msg=(
                        "grouped cpp wrapper mismatch for "
                        f"sequence_length={COMPILE_SEQUENCE_LENGTH}"
                    ),
                )
                self.assertIn("'group_enabled': True", cpp_code)
                self.assertIn("switch (grouped_group_id)", cpp_code)
                self.assertGreaterEqual(
                    cpp_code.count("static void* grouped_kernel_v"), 2
                )

                for sequence_length, inputs in (
                    (4, _make_gated_transpose_inputs(4)),
                    (256, _make_gated_transpose_inputs(256)),
                ):
                    _mark_sequence_length_dynamic(inputs)
                    with torch.no_grad():
                        expected = model(*inputs)
                        actual = compiled(*inputs)
                        torch.npu.synchronize()
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=0.02,
                        atol=0.03,
                        msg=(
                            "grouped cpp wrapper mismatch for "
                            f"sequence_length={sequence_length}"
                        ),
                    )
        finally:
            npu_config.enable_symbolic_shape_group_autotune = (
                previous_group_autotune
            )


if __name__ == "__main__":
    run_tests()
