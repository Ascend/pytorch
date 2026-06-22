import json
import os
import subprocess
import sys
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.virtualized import V
from torch_npu._inductor._aclgraph_update_plan import (
    ACLGRAPH_UPDATE_PLAN_GLOBAL,
    append_inductor_aclgraph_update_plan_for_codegen_node,
)
from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.codegen.wrapper import (
    NpuMlirSubgraphPythonWrapperCodegen,
    NpuMlirWrapperCodeGen,
)
from torch_npu._inductor.codegen.wrapper import (
    _NPUKernelCodegenMixin,
    NPUSubgraphPythonWrapperCodegen,
)
from torch.testing._internal.common_utils import run_tests

import torch_npu
import torch_npu._inductor
from torch_npu.testing.common_utils import SupportedDevices

from testutils import TestUtils


def _make_ifa_inputs():
    q = torch.randn(1, 32, 1, 128, dtype=torch.float16, device="npu")
    k = torch.randn(1, 32, 1, 128, dtype=torch.float16, device="npu")
    v = torch.randn(1, 32, 1, 128, dtype=torch.float16, device="npu")
    return q, k, v


def _ifa_with_const_actual_seq_lengths(query, key, value):
    out, _ = torch_npu.npu_fused_infer_attention_score(
        query,
        key,
        value,
        num_heads=32,
        input_layout="BNSD",
        scale=128.0,
        pre_tokens=65535,
        next_tokens=65535,
        softmax_lse_flag=False,
        actual_seq_lengths=[37],
    )
    return out


def _ifa_with_actual_seq_lengths_kv(query, key, value):
    out, _ = torch_npu.npu_fused_infer_attention_score(
        query,
        key,
        value,
        num_heads=32,
        input_layout="BNSD",
        scale=128.0,
        pre_tokens=65535,
        next_tokens=65535,
        softmax_lse_flag=False,
        actual_seq_lengths=[37],
        actual_seq_lengths_kv=[1],
    )
    return out


def _ifa_with_runtime_actual_seq_lengths(query, key, value, actual_seq_lengths):
    out, _ = torch_npu.npu_fused_infer_attention_score(
        query,
        key,
        value,
        num_heads=32,
        input_layout="BNSD",
        scale=128.0,
        pre_tokens=65535,
        next_tokens=65535,
        softmax_lse_flag=False,
        actual_seq_lengths=actual_seq_lengths,
    )
    return out


def _ifa_v2_with_const_actual_seq_qlen(query, key, value):
    out, _ = torch_npu.npu_fused_infer_attention_score_v2(
        query,
        key,
        value,
        num_query_heads=32,
        input_layout="BNSD",
        softmax_scale=128.0,
        pre_tokens=65535,
        next_tokens=65535,
        return_softmax_lse=False,
        actual_seq_qlen=[1],
    )
    return out


def _two_ifa_with_const_actual_seq_lengths(query, key, value):
    out1, _ = torch_npu.npu_fused_infer_attention_score(
        query,
        key,
        value,
        num_heads=32,
        input_layout="BNSD",
        scale=128.0,
        pre_tokens=65535,
        next_tokens=65535,
        softmax_lse_flag=False,
        actual_seq_lengths=[37],
    )
    out2, _ = torch_npu.npu_fused_infer_attention_score(
        query,
        key,
        out1,
        num_heads=32,
        input_layout="BNSD",
        scale=128.0,
        pre_tokens=65535,
        next_tokens=65535,
        softmax_lse_flag=False,
        actual_seq_lengths=[41],
    )
    return out2


def _run_and_get_code_without_reset(fn, *args):
    from torch._inductor.graph import GraphLowering

    source_codes = []

    def save_output_code(code):
        source_codes.append(code)

    with mock.patch.object(GraphLowering, "save_output_code", save_output_code):
        result = fn(*args)
    return result, source_codes


def _compiled_code(
    fn,
    *args,
    cudagraphs=True,
    cudagraph_trees=True,
    graph_partition=False,
    npu_backend=None,
):
    torch._dynamo.reset()
    old_cudagraphs = config.triton.cudagraphs
    old_cudagraph_trees = config.triton.cudagraph_trees
    old_force_disable_caches = config.force_disable_caches
    old_graph_partition = config.graph_partition
    try:
        config.triton.cudagraphs = cudagraphs
        config.triton.cudagraph_trees = cudagraph_trees
        config.force_disable_caches = True
        config.graph_partition = graph_partition
        options = {"npu_backend": npu_backend} if npu_backend is not None else None
        compiled = torch.compile(
            fn,
            backend="inductor",
            fullgraph=True,
            options=options,
        )
        _, codes = _run_and_get_code_without_reset(compiled, *args)
    finally:
        config.triton.cudagraphs = old_cudagraphs
        config.triton.cudagraph_trees = old_cudagraph_trees
        config.force_disable_caches = old_force_disable_caches
        config.graph_partition = old_graph_partition
        torch._dynamo.reset()
    return "\n".join(codes)


def _compiled_code_with_backend_in_subprocess(npu_backend):
    script = f"""
import json
import torch

from test_aclgraph_update_plan_compile import (
    _compiled_code,
    _ifa_with_const_actual_seq_lengths,
    _make_ifa_inputs,
)

torch.npu.set_device(0)
code = _compiled_code(
    _ifa_with_const_actual_seq_lengths,
    *_make_ifa_inputs(),
    npu_backend={npu_backend!r},
)
print("ACLGRAPH_CODE_BEGIN")
print(json.dumps(code))
print("ACLGRAPH_CODE_END")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise AssertionError(
            f"{npu_backend} subprocess compile failed.\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    begin = "ACLGRAPH_CODE_BEGIN"
    end = "ACLGRAPH_CODE_END"
    if begin not in result.stdout or end not in result.stdout:
        raise AssertionError(
            f"Failed to collect generated code from {npu_backend} subprocess.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    payload = result.stdout.split(begin, 1)[1].split(end, 1)[0].strip()
    return json.loads(payload)


class TestACLGraphUpdatePlanCompile(TestUtils):

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_ifa_no_graph_partition_codegen_attaches_plan_to_call_function(self):
        torch.npu.set_device(0)
        torch._dynamo.reset()

        code = _compiled_code(
            _ifa_with_const_actual_seq_lengths,
            *_make_ifa_inputs(),
            graph_partition=False,
        )

        self.assertIn("_torch_npu_aclgraph_update_plan", code)
        self.assertIn("def call(args):", code)
        self.assertIn(f"call.{ACLGRAPH_UPDATE_PLAN_GLOBAL}", code)
        self.assertEqual(code.count(f"call.{ACLGRAPH_UPDATE_PLAN_GLOBAL}"), 1)
        self.assertNotIn("def partition_0(args):", code)
        self.assertNotIn(f"partition_0.{ACLGRAPH_UPDATE_PLAN_GLOBAL}", code)
        self.assertNotIn(f"\n{ACLGRAPH_UPDATE_PLAN_GLOBAL} = ", code)
        self.assertIn("npu_fused_infer_attention_score.default", code)
        self.assertIn("actual_seq_lengths", code)
        self.assertIn("'value': 37", code)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_ifa_codegen_plan_survives_generated_code_reload(self):
        torch.npu.set_device(0)
        torch._dynamo.reset()

        code = _compiled_code(_ifa_with_const_actual_seq_lengths, *_make_ifa_inputs())
        namespace = {}
        exec(compile(code, "<aclgraph_update_plan_test>", "exec"), namespace)

        plan = getattr(namespace["call"], ACLGRAPH_UPDATE_PLAN_GLOBAL)
        self.assertEqual(plan[0]["op"], "npu_fused_infer_attention_score.default")
        self.assertEqual(
            plan[0]["updates"]["actual_seq_lengths"],
            {
                "kind": "list",
                "items": [{"kind": "constant", "value": 37}],
            },
        )
        if "actual_seq_lengths_kv" in plan[0]["updates"]:
            self.assertEqual(plan[0]["updates"]["actual_seq_lengths_kv"], {"kind": "none"})

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_ifa_codegen_emits_multiple_actual_seq_keys(self):
        torch.npu.set_device(0)
        torch._dynamo.reset()

        code = _compiled_code(_ifa_with_actual_seq_lengths_kv, *_make_ifa_inputs())

        self.assertIn("actual_seq_lengths", code)
        self.assertIn("actual_seq_lengths_kv", code)
        self.assertIn("'value': 37", code)
        self.assertIn("'value': 1", code)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_ifa_codegen_recompiles_when_guarded_actual_seq_list_changes(self):
        torch.npu.set_device(0)
        torch._dynamo.reset()

        old_cudagraphs = config.triton.cudagraphs
        old_cudagraph_trees = config.triton.cudagraph_trees
        old_force_disable_caches = config.force_disable_caches
        try:
            config.triton.cudagraphs = True
            config.triton.cudagraph_trees = True
            config.force_disable_caches = True
            compiled = torch.compile(
                _ifa_with_runtime_actual_seq_lengths,
                backend="inductor",
                fullgraph=True,
            )
            code_37 = "\n".join(
                _run_and_get_code_without_reset(compiled, *_make_ifa_inputs(), [37])[1]
            )
            code_41 = "\n".join(
                _run_and_get_code_without_reset(compiled, *_make_ifa_inputs(), [41])[1]
            )
        finally:
            config.triton.cudagraphs = old_cudagraphs
            config.triton.cudagraph_trees = old_cudagraph_trees
            config.force_disable_caches = old_force_disable_caches
            torch._dynamo.reset()

        self.assertIn("'value': 37", code_37)
        self.assertNotIn("'value': 37", code_41)
        self.assertTrue(
            "'value': 41" in code_41
            or ("'kind': 'input'" in code_41 and "'index': 3" in code_41)
        )

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_ifa_v2_codegen_emits_aclgraph_update_plan(self):
        torch.npu.set_device(0)
        torch._dynamo.reset()

        code = _compiled_code(_ifa_v2_with_const_actual_seq_qlen, *_make_ifa_inputs())

        self.assertIn("_torch_npu_aclgraph_update_plan", code)
        self.assertIn("npu_fused_infer_attention_score_v2.default", code)
        self.assertIn("actual_seq_qlen", code)
        self.assertIn("'value': 1", code)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_ifa_graph_partition_codegen_attaches_plan_to_partition_function(self):
        class Graph:
            cpp_wrapper = False
            disable_cudagraphs_reason = None

        plan = [
            {
                "op": "npu_fused_infer_attention_score.default",
                "updates": {
                    "actual_seq_lengths": {
                        "kind": "list",
                        "items": [{"kind": "constant", "value": 37}],
                    }
                },
            }
        ]

        old_cudagraphs = config.triton.cudagraphs
        old_cudagraph_trees = config.triton.cudagraph_trees
        old_graph_partition = config.graph_partition
        try:
            config.triton.cudagraphs = True
            config.triton.cudagraph_trees = True
            config.graph_partition = True

            wrapper = object.__new__(NPUSubgraphPythonWrapperCodegen)
            wrapper.launcher_fn_name = "partition_0"
            with V.set_graph_handler(Graph()):
                wrapper.torch_npu_aclgraph_update_plan = plan
                result = IndentedBuffer()
                wrapper.generate_after_suffix(result)
        finally:
            config.triton.cudagraphs = old_cudagraphs
            config.triton.cudagraph_trees = old_cudagraph_trees
            config.graph_partition = old_graph_partition

        code = result.getvalue()
        self.assertIn(f"partition_0.{ACLGRAPH_UPDATE_PLAN_GLOBAL}", code)
        self.assertEqual(code.count(f"partition_0.{ACLGRAPH_UPDATE_PLAN_GLOBAL}"), 1)
        self.assertNotIn(f"call.{ACLGRAPH_UPDATE_PLAN_GLOBAL}", code)
        self.assertNotIn(f"\n{ACLGRAPH_UPDATE_PLAN_GLOBAL} = ", code)
        self.assertIn(repr(plan), code)

    def test_mlir_dvm_wrapper_appends_aclgraph_update_plan_for_extern_kernel(self):
        class Graph:
            cpp_wrapper = False
            disable_cudagraphs_reason = None

        class Arg:
            def __init__(self, name):
                self.name = name

        class Schema:
            arguments = [
                Arg("query"),
                Arg("key"),
                Arg("value"),
                Arg("num_heads"),
                Arg("input_layout"),
                Arg("actual_seq_lengths"),
            ]

        class Target:
            __name__ = "npu_fused_infer_attention_score.default"
            _schema = Schema()

        class Value:
            def __init__(self, name):
                self.name = name

            def get_name(self):
                return self.name

        class Kernel:
            op_overload = Target()
            inputs = [Value("arg0_1"), Value("arg1_1"), Value("arg2_1")]
            constant_args = [32, "BNSD", [37]]
            kwargs = {}
            layout = object()

            def get_name(self):
                return "buf0"

            def get_origin_node(self):
                return None

            def get_kernel_name(self):
                return "torch.ops.npu.npu_fused_infer_attention_score.default"

        old_cudagraphs = config.triton.cudagraphs
        old_cudagraph_trees = config.triton.cudagraph_trees
        old_graph_partition = config.graph_partition
        try:
            config.triton.cudagraphs = True
            config.triton.cudagraph_trees = True
            config.graph_partition = False

            wrapper = object.__new__(NpuMlirWrapperCodeGen)
            wrapper.launcher_fn_name = "call"
            wrapper.declare = ""
            wrapper.ending = ""
            wrapper.supports_intermediate_hooks = False
            wrapper.get_graph_input_names = lambda: ["arg0_1", "arg1_1", "arg2_1"]
            wrapper.get_graph_inputs = lambda: {}
            wrapper.writeline = lambda line: None

            with V.set_graph_handler(Graph()):
                append_inductor_aclgraph_update_plan_for_codegen_node(wrapper, Kernel())
                result = IndentedBuffer()
                wrapper.generate_after_suffix(result)
        finally:
            config.triton.cudagraphs = old_cudagraphs
            config.triton.cudagraph_trees = old_cudagraph_trees
            config.graph_partition = old_graph_partition

        self.assertEqual(
            wrapper.torch_npu_aclgraph_update_plan,
            [
                {
                    "op": "npu_fused_infer_attention_score.default",
                    "updates": {
                        "actual_seq_lengths": {
                            "kind": "list",
                            "items": [{"kind": "constant", "value": 37}],
                        }
                    },
                }
            ],
        )
        code = result.getvalue()
        self.assertIn(f"call.{ACLGRAPH_UPDATE_PLAN_GLOBAL}", code)
        self.assertIn(repr(wrapper.torch_npu_aclgraph_update_plan), code)

    def test_mlir_dvm_subgraph_wrapper_emits_aclgraph_update_plan(self):
        class Graph:
            cpp_wrapper = False
            disable_cudagraphs_reason = None

        plan = [
            {
                "op": "npu_fused_infer_attention_score.default",
                "updates": {
                    "actual_seq_lengths": {
                        "kind": "list",
                        "items": [{"kind": "constant", "value": 37}],
                    }
                },
            }
        ]

        old_cudagraphs = config.triton.cudagraphs
        old_cudagraph_trees = config.triton.cudagraph_trees
        old_graph_partition = config.graph_partition
        try:
            config.triton.cudagraphs = True
            config.triton.cudagraph_trees = True
            config.graph_partition = True

            with V.set_graph_handler(Graph()):
                wrapper = object.__new__(NpuMlirSubgraphPythonWrapperCodegen)
                wrapper.launcher_fn_name = "partition_0"
                wrapper.subgraph_name = "partition_0"
                wrapper.torch_npu_aclgraph_update_plan = plan
                result = IndentedBuffer()
                wrapper.generate_after_suffix(result)
        finally:
            config.triton.cudagraphs = old_cudagraphs
            config.triton.cudagraph_trees = old_cudagraph_trees
            config.graph_partition = old_graph_partition

        code = result.getvalue()
        self.assertIn(f"partition_0.{ACLGRAPH_UPDATE_PLAN_GLOBAL}", code)
        self.assertNotIn(f"call.{ACLGRAPH_UPDATE_PLAN_GLOBAL}", code)
        self.assertIn(repr(plan), code)

    def test_mlir_dvm_wrapper_does_not_inherit_default_npu_codegen_mixin(self):
        self.assertFalse(issubclass(NpuMlirWrapperCodeGen, _NPUKernelCodegenMixin))
        self.assertFalse(issubclass(NpuMlirSubgraphPythonWrapperCodegen, _NPUKernelCodegenMixin))

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_ifa_codegen_preserves_multiple_plan_entry_order(self):
        torch.npu.set_device(0)
        torch._dynamo.reset()

        code = _compiled_code(
            _two_ifa_with_const_actual_seq_lengths,
            *_make_ifa_inputs(),
        )

        self.assertGreaterEqual(
            code.count("'op': 'npu_fused_infer_attention_score.default'"),
            2,
        )
        first_update = code.find("'value': 37")
        second_update = code.find("'value': 41")
        self.assertGreaterEqual(first_update, 0)
        self.assertGreater(second_update, first_update)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_ifa_codegen_skips_aclgraph_update_plan_without_cudagraphs(self):
        torch.npu.set_device(0)
        torch._dynamo.reset()

        code = _compiled_code(
            _ifa_with_const_actual_seq_lengths,
            *_make_ifa_inputs(),
            cudagraphs=False,
        )

        self.assertNotIn("_torch_npu_aclgraph_update_plan", code)

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_ifa_codegen_skips_aclgraph_update_plan_without_cudagraph_trees(self):
        torch.npu.set_device(0)
        torch._dynamo.reset()

        code = _compiled_code(
            _ifa_with_const_actual_seq_lengths,
            *_make_ifa_inputs(),
            cudagraph_trees=False,
        )

        self.assertNotIn("_torch_npu_aclgraph_update_plan", code)

    def test_wrapper_plan_gate_respects_cudagraph_disable_reason(self):
        from torch._inductor.virtualized import V
        from torch_npu._inductor._aclgraph_update_plan.codegen import (
            should_generate_inductor_aclgraph_update_plan,
        )

        class Graph:
            cpp_wrapper = False
            disable_cudagraphs_reason = "unsupported"

        old_cudagraphs = config.triton.cudagraphs
        old_cudagraph_trees = config.triton.cudagraph_trees
        try:
            config.triton.cudagraphs = True
            config.triton.cudagraph_trees = True
            with V.set_graph_handler(Graph()):
                self.assertFalse(should_generate_inductor_aclgraph_update_plan())
        finally:
            config.triton.cudagraphs = old_cudagraphs
            config.triton.cudagraph_trees = old_cudagraph_trees

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_ifa_cudagraph_tree_receives_aclgraph_update_plan(self):
        torch.npu.set_device(0)
        torch._dynamo.reset()

        import torch_npu.npu._graph_tree as graph_tree

        old_cudagraphs = config.triton.cudagraphs
        old_cudagraph_trees = config.triton.cudagraph_trees
        old_force_disable_caches = config.force_disable_caches
        old_slow_path_asserts = config.triton.slow_path_cudagraph_asserts
        original_update = graph_tree.update_aclgraph_records_for_graph
        seen_plans = []

        def collect_plan(plan, graph, inputs):
            seen_plans.append(plan)
            return original_update(plan, graph, inputs)

        try:
            config.triton.cudagraphs = True
            config.triton.cudagraph_trees = True
            config.force_disable_caches = True
            config.triton.slow_path_cudagraph_asserts = False
            graph_tree.update_aclgraph_records_for_graph = collect_plan

            compiled = torch.compile(
                _ifa_with_const_actual_seq_lengths,
                backend="inductor",
                fullgraph=True,
            )
            inputs = _make_ifa_inputs()
            expected = _ifa_with_const_actual_seq_lengths(*inputs)
            actual = compiled(*inputs)
            torch.testing.assert_close(
                actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3
            )

            inputs = _make_ifa_inputs()
            expected = _ifa_with_const_actual_seq_lengths(*inputs)
            actual = compiled(*inputs)
            torch.testing.assert_close(
                actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3
            )
        finally:
            graph_tree.update_aclgraph_records_for_graph = original_update
            config.triton.cudagraphs = old_cudagraphs
            config.triton.cudagraph_trees = old_cudagraph_trees
            config.force_disable_caches = old_force_disable_caches
            config.triton.slow_path_cudagraph_asserts = old_slow_path_asserts
            torch._dynamo.reset()

        self.assertTrue(seen_plans)
        self.assertTrue(any(plan for plan in seen_plans))
        plan = next(plan for plan in seen_plans if plan)
        self.assertEqual(plan[0]["op"], "npu_fused_infer_attention_score.default")
        self.assertEqual(
            plan[0]["updates"]["actual_seq_lengths"],
            {"kind": "list", "items": [{"kind": "constant", "value": 37}]},
        )

    def test_npugraphify_keeps_aclgraph_update_plan_on_callable_attribute(self):
        import torch_npu.npu._graph_tree as graph_tree

        expected_plan = [{"op": "test.op", "updates": {}}]

        def model(args):
            return args

        setattr(model, ACLGRAPH_UPDATE_PLAN_GLOBAL, expected_plan)

        captured = {}

        def fake_add_function(*args, **kwargs):
            captured["arg_count"] = len(args)
            captured["model"] = args[0]
            return lambda inputs: inputs, []

        manager = mock.Mock()
        manager.add_function.side_effect = fake_add_function
        with mock.patch(
            "torch_npu.npu._graph_tree.get_container",
            return_value=mock.Mock(get_tree_manager=mock.Mock(return_value=manager)),
        ):
            graph_tree.npugraphify(
                model,
                [],
                device_index=0,
                is_backward=False,
                is_inference=True,
            )

        self.assertEqual(captured["arg_count"], 8)
        self.assertIs(
            getattr(captured["model"], ACLGRAPH_UPDATE_PLAN_GLOBAL),
            expected_plan,
        )

    @SupportedDevices(["Ascend910B", "Ascend910_93"])
    def test_ifa_v2_cudagraph_tree_receives_aclgraph_update_plan(self):
        torch.npu.set_device(0)
        torch._dynamo.reset()

        import torch_npu.npu._graph_tree as graph_tree

        old_cudagraphs = config.triton.cudagraphs
        old_cudagraph_trees = config.triton.cudagraph_trees
        old_force_disable_caches = config.force_disable_caches
        old_slow_path_asserts = config.triton.slow_path_cudagraph_asserts
        original_update = graph_tree.update_aclgraph_records_for_graph
        seen_plans = []

        def collect_plan(plan, graph, inputs):
            seen_plans.append(plan)
            return original_update(plan, graph, inputs)

        try:
            config.triton.cudagraphs = True
            config.triton.cudagraph_trees = True
            config.force_disable_caches = True
            config.triton.slow_path_cudagraph_asserts = False
            graph_tree.update_aclgraph_records_for_graph = collect_plan

            compiled = torch.compile(
                _ifa_v2_with_const_actual_seq_qlen,
                backend="inductor",
                fullgraph=True,
            )
            inputs = _make_ifa_inputs()
            expected = _ifa_v2_with_const_actual_seq_qlen(*inputs)
            actual = compiled(*inputs)
            torch.testing.assert_close(
                actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3
            )

            inputs = _make_ifa_inputs()
            expected = _ifa_v2_with_const_actual_seq_qlen(*inputs)
            actual = compiled(*inputs)
            torch.testing.assert_close(
                actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3
            )
        finally:
            graph_tree.update_aclgraph_records_for_graph = original_update
            config.triton.cudagraphs = old_cudagraphs
            config.triton.cudagraph_trees = old_cudagraph_trees
            config.force_disable_caches = old_force_disable_caches
            config.triton.slow_path_cudagraph_asserts = old_slow_path_asserts
            torch._dynamo.reset()

        self.assertTrue(any(plan for plan in seen_plans))
        plan = next(plan for plan in seen_plans if plan)
        self.assertEqual(plan[0]["op"], "npu_fused_infer_attention_score_v2.default")
        self.assertEqual(
            plan[0]["updates"]["actual_seq_qlen"],
            {"kind": "list", "items": [{"kind": "constant", "value": 1}]},
        )


if __name__ == "__main__":
    run_tests()
