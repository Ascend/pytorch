import unittest

from torch_npu.npu._aclgraph_update_plan import (
    ACLGRAPH_UPDATE_PLAN_GLOBAL,
    resolve_aclgraph_update_plan,
    validate_aclgraph_update_plan,
)
from torch_npu._inductor._aclgraph_update_plan.codegen import (
    build_aclgraph_update_plan_entry_for_inductor,
)
from torch_npu.npu._aclgraph_update_plan.resolver import (
    build_cpu_update_input_for_graph,
)


class TestACLGraphUpdatePlan(unittest.TestCase):
    def setUp(self):
        from torch_npu.npu._npugraph_handlers.npugraph_handler import _NPU_GRAPH_OP_HANDLERS

        class Handler:
            UPDATE_SPECS = {
                "npu_fusion_attention_v3.default": [
                    ("arg", 14, "actual_seq_qlen"),
                    ("arg", 15, "actual_seq_kvlen"),
                ],
                "npu_fusion_attention_v3.out": [
                    ("arg", 14, "actual_seq_qlen"),
                    ("arg", 15, "actual_seq_kvlen"),
                ],
                "npu_fused_infer_attention_score.default": [
                    ("arg", 5, "actual_seq_lengths"),
                    ("arg", 6, "actual_seq_lengths_kv"),
                ],
                "npu_fused_infer_attention_score_v2.default": [
                    ("arg", 7, "actual_seq_qlen"),
                    ("arg", 8, "actual_seq_kvlen"),
                ],
            }

            @classmethod
            def get_update_specs(cls, op_name):
                return cls.UPDATE_SPECS.get(op_name, [])

        self._old_handlers = dict(_NPU_GRAPH_OP_HANDLERS)
        _NPU_GRAPH_OP_HANDLERS.update({
            "npu_fusion_attention_v3.default": Handler,
            "npu_fusion_attention_v3.out": Handler,
            "npu_fused_infer_attention_score.default": Handler,
            "npu_fused_infer_attention_score_v2.default": Handler,
        })

    def tearDown(self):
        from torch_npu.npu._npugraph_handlers.npugraph_handler import _NPU_GRAPH_OP_HANDLERS

        _NPU_GRAPH_OP_HANDLERS.clear()
        _NPU_GRAPH_OP_HANDLERS.update(self._old_handlers)

    def test_build_inductor_plan_maps_graph_input_sources(self):
        class Arg:
            def __init__(self, name):
                self.name = name

        class Schema:
            arguments = [
                Arg("query"),
                Arg("key"),
                Arg("value"),
                Arg("head_num"),
                Arg("input_layout"),
                Arg("pse"),
                Arg("padding_mask"),
                Arg("atten_mask"),
                Arg("scale"),
                Arg("keep_prob"),
                Arg("pre_tockens"),
                Arg("next_tockens"),
                Arg("inner_precise"),
                Arg("prefix"),
                Arg("actual_seq_qlen"),
                Arg("actual_seq_kvlen"),
            ]

        class Target:
            __name__ = "npu_fusion_attention_v3.default"
            _schema = Schema()

        class Value:
            def __init__(self, name):
                self.name = name

            def get_name(self):
                return self.name

        actual = Value("arg0_1")
        q = Value("arg1_1")
        k = Value("arg2_1")
        v = Value("arg3_1")

        self.assertEqual(
            build_aclgraph_update_plan_entry_for_inductor(
                Target(),
                (
                    q, k, v, 1, "TND", None, None, None, 1.0, 1.0,
                    2147483647, 2147483647, 0, None, actual, actual,
                ),
                {},
                ["arg0_1", "arg1_1", "arg2_1", "arg3_1"],
                {},
            ),
            {
                "op": "npu_fusion_attention_v3.default",
                "updates": {
                    "actual_seq_qlen": {"kind": "input", "index": 0},
                    "actual_seq_kvlen": {"kind": "input", "index": 0},
                },
            },
        )

    def test_build_inductor_plan_skips_fa3_bnsd_like_runtime_handler(self):
        class Arg:
            def __init__(self, name):
                self.name = name

        class Schema:
            arguments = [
                Arg("query"),
                Arg("key"),
                Arg("value"),
                Arg("head_num"),
                Arg("input_layout"),
                Arg("pse"),
                Arg("padding_mask"),
                Arg("atten_mask"),
                Arg("scale"),
                Arg("keep_prob"),
                Arg("pre_tockens"),
                Arg("next_tockens"),
                Arg("inner_precise"),
                Arg("prefix"),
                Arg("actual_seq_qlen"),
                Arg("actual_seq_kvlen"),
            ]

        class Target:
            __name__ = "npu_fusion_attention_v3.default"
            _schema = Schema()

        class Value:
            def __init__(self, name):
                self.name = name

            def get_name(self):
                return self.name

        actual = Value("arg0_1")
        q = Value("arg1_1")
        k = Value("arg2_1")
        v = Value("arg3_1")
        args_prefix = (q, k, v, 1)
        args_suffix = (
            None, None, None, 1.0, 1.0,
            2147483647, 2147483647, 0, None, actual, actual,
        )

        self.assertIsNone(
            build_aclgraph_update_plan_entry_for_inductor(
                Target(),
                args_prefix + ("BNSD",) + args_suffix,
                {},
                ["arg0_1", "arg1_1", "arg2_1", "arg3_1"],
                {},
            )
        )
        self.assertEqual(
            build_aclgraph_update_plan_entry_for_inductor(
                Target(),
                args_prefix + ("TND",) + args_suffix,
                {},
                ["arg0_1", "arg1_1", "arg2_1", "arg3_1"],
                {},
            ),
            {
                "op": "npu_fusion_attention_v3.default",
                "updates": {
                    "actual_seq_qlen": {"kind": "input", "index": 0},
                    "actual_seq_kvlen": {"kind": "input", "index": 0},
                },
            },
        )

    def test_build_inductor_plan_ignores_unhandled_ops(self):
        class Arg:
            def __init__(self, name):
                self.name = name

        class Schema:
            arguments = [Arg("actual_seq_qlen")]

        class Target:
            __name__ = "unhandled_attention.default"
            _schema = Schema()

        self.assertIsNone(
            build_aclgraph_update_plan_entry_for_inductor(
                Target(),
                [4],
                {},
                [],
                {},
            )
        )

    def test_build_inductor_plan_filters_to_handler_update_specs(self):
        from torch_npu.npu._npugraph_handlers.npugraph_handler import _NPU_GRAPH_OP_HANDLERS

        class Handler:
            @classmethod
            def get_update_specs(cls, op_name):
                return [("arg", 6, "actual_seq_lengths_kv")]

        class Arg:
            def __init__(self, name):
                self.name = name

        class Schema:
            arguments = [
                Arg("query"),
                Arg("key"),
                Arg("value"),
                Arg("pse_shift"),
                Arg("atten_mask"),
                Arg("actual_seq_lengths"),
                Arg("actual_seq_lengths_kv"),
            ]

        class Target:
            __name__ = "npu_fused_infer_attention_score.default"
            _schema = Schema()

        _NPU_GRAPH_OP_HANDLERS["npu_fused_infer_attention_score.default"] = Handler
        self.assertEqual(
            build_aclgraph_update_plan_entry_for_inductor(
                Target(),
                ["q", "k", "v", None, None, [15], [100]],
                {},
                [],
                {},
            ),
            {
                "op": "npu_fused_infer_attention_score.default",
                "updates": {
                    "actual_seq_lengths_kv": {"kind": "list", "items": [
                        {"kind": "constant", "value": 100},
                    ]},
                },
            },
        )

    def test_build_inductor_plan_for_ifa_v1_positional_actual_seq_lengths(self):
        class Arg:
            def __init__(self, name):
                self.name = name

        class Schema:
            arguments = [
                Arg("query"),
                Arg("key"),
                Arg("value"),
                Arg("pse_shift"),
                Arg("atten_mask"),
                Arg("actual_seq_lengths"),
                Arg("actual_seq_lengths_kv"),
            ]

        class Target:
            __name__ = "npu_fused_infer_attention_score.default"
            _schema = Schema()

        self.assertEqual(
            build_aclgraph_update_plan_entry_for_inductor(
                Target(),
                ["q", "k", "v", None, None, [15], [100]],
                {},
                [],
                {},
            ),
            {
                "op": "npu_fused_infer_attention_score.default",
                "updates": {
                    "actual_seq_lengths": {"kind": "list", "items": [
                        {"kind": "constant", "value": 15},
                    ]},
                    "actual_seq_lengths_kv": {"kind": "list", "items": [
                        {"kind": "constant", "value": 100},
                    ]},
                },
            },
        )

    def test_build_inductor_plan_for_ifa_v2_positional_actual_seq_qlen(self):
        class Arg:
            def __init__(self, name):
                self.name = name

        class Schema:
            arguments = [
                Arg("query"),
                Arg("key"),
                Arg("value"),
                Arg("query_rope"),
                Arg("key_rope"),
                Arg("pse_shift"),
                Arg("atten_mask"),
                Arg("actual_seq_qlen"),
                Arg("actual_seq_kvlen"),
            ]

        class Target:
            __name__ = "npu_fused_infer_attention_score_v2.default"
            _schema = Schema()

        self.assertEqual(
            build_aclgraph_update_plan_entry_for_inductor(
                Target(),
                ["q", "k", "v", None, None, None, None, [16], [128]],
                {},
                [],
                {},
            ),
            {
                "op": "npu_fused_infer_attention_score_v2.default",
                "updates": {
                    "actual_seq_qlen": {"kind": "list", "items": [
                        {"kind": "constant", "value": 16},
                    ]},
                    "actual_seq_kvlen": {"kind": "list", "items": [
                        {"kind": "constant", "value": 128},
                    ]},
                },
            },
        )

    def test_build_inductor_plan_rejects_unlifted_tensor_actual_seq_constant(self):
        import torch

        class Arg:
            def __init__(self, name):
                self.name = name

        class Schema:
            arguments = [
                Arg("query"),
                Arg("key"),
                Arg("value"),
                Arg("pse_shift"),
                Arg("atten_mask"),
                Arg("actual_seq_lengths"),
            ]

        class Target:
            __name__ = "npu_fused_infer_attention_score.default"
            _schema = Schema()

        with self.assertRaisesRegex(RuntimeError, "Tensor constant"):
            build_aclgraph_update_plan_entry_for_inductor(
                Target(),
                ["q", "k", "v", None, None, torch.tensor([15])],
                {},
                [],
                {},
            )

    def test_resolve_input_and_constant_sources(self):
        new_inputs = ["q", "qlen", "kvlen"]
        plan = [
            {
                "op": "npu_fusion_attention_v3.out",
                "updates": {
                    "actual_seq_qlen": {"kind": "input", "index": 1},
                    "actual_seq_kvlen": {"kind": "list", "items": [
                        {"kind": "constant", "value": 4},
                    ]},
                },
            }
        ]

        self.assertEqual(ACLGRAPH_UPDATE_PLAN_GLOBAL, "_torch_npu_aclgraph_update_plan")
        self.assertEqual(
            resolve_aclgraph_update_plan(plan, new_inputs),
            [{"actual_seq_qlen": "qlen", "actual_seq_kvlen": [4]}],
        )

    def test_resolve_list_source_with_input_and_constant_items(self):
        new_inputs = ["q", 10000]
        plan = [
            {
                "op": "npu_fused_infer_attention_score.default",
                "updates": {
                    "actual_seq_lengths": {"kind": "list", "items": [
                        {"kind": "constant", "value": 15},
                        {"kind": "input", "index": 1},
                    ]},
                    "actual_seq_lengths_kv": {"kind": "list", "items": [
                        {"kind": "input", "index": 1},
                    ]},
                },
            }
        ]

        self.assertEqual(
            resolve_aclgraph_update_plan(plan, new_inputs),
            [{"actual_seq_lengths": [15, 10000], "actual_seq_lengths_kv": [10000]}],
        )

    def test_resolve_rejects_out_of_range_input_index(self):
        plan = [
            {
                "op": "npu_fusion_attention_v3.out",
                "updates": {
                    "actual_seq_qlen": {"kind": "input", "index": 3},
                },
            }
        ]

        with self.assertRaisesRegex(RuntimeError, "out of range"):
            resolve_aclgraph_update_plan(plan, ["only_one_input"])

    def test_validate_plan_rejects_length_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "length mismatch"):
            validate_aclgraph_update_plan(
                [{"op": "npu_fusion_attention_v3.out", "updates": {}}],
                [],
            )

    def test_validate_plan_rejects_missing_plan_with_cache_hint(self):
        class Record:
            class Op:
                __name__ = "npu_fused_infer_attention_score.default"

            op_cache_entry = Op()
            kwargs = {"actual_seq_lengths": None}

        with self.assertRaisesRegex(RuntimeError, "cached compiled code"):
            validate_aclgraph_update_plan([], [Record()])

    def test_validate_plan_rejects_op_mismatch(self):
        class Record:
            class Op:
                __name__ = "npu_fused_infer_attention_score.out"

            op_cache_entry = Op()
            kwargs = {}

        with self.assertRaisesRegex(RuntimeError, "op mismatch"):
            validate_aclgraph_update_plan(
                [{"op": "npu_fusion_attention_v3.out", "updates": {}}],
                [Record()],
            )

    def test_validate_plan_rejects_invalid_entry_shape(self):
        class Record:
            class Op:
                __name__ = "npu_fusion_attention_v3.out"

            op_cache_entry = Op()
            kwargs = {"actual_seq_qlen": None}

        with self.assertRaisesRegex(RuntimeError, "invalid plan entry"):
            validate_aclgraph_update_plan(
                [{"updates": {
                    "actual_seq_qlen": {"kind": "constant", "value": 4},
                }}],
                [Record()],
            )

    def test_validate_plan_rejects_invalid_source_shape(self):
        class Record:
            class Op:
                __name__ = "npu_fusion_attention_v3.out"

            op_cache_entry = Op()
            kwargs = {"actual_seq_qlen": None}

        with self.assertRaisesRegex(RuntimeError, "invalid source"):
            validate_aclgraph_update_plan(
                [{"op": "npu_fusion_attention_v3.out", "updates": {
                    "actual_seq_qlen": "not_a_source",
                }}],
                [Record()],
            )

    def test_validate_plan_allows_default_out_compatibility(self):
        class Record:
            class Op:
                __name__ = "npu_fusion_attention_v3.out"

            op_cache_entry = Op()
            kwargs = {"actual_seq_qlen": None}

        validate_aclgraph_update_plan(
            [{"op": "npu_fusion_attention_v3.default", "updates": {
                "actual_seq_qlen": {"kind": "list", "items": [
                    {"kind": "constant", "value": 4},
                ]},
            }}],
            [Record()],
        )

    def test_validate_plan_reads_legacy_handler_update_specs(self):
        from torch_npu.npu._npugraph_handlers.npugraph_handler import _NPU_GRAPH_OP_HANDLERS

        class Handler:
            UPDATE_SPECS = {
                "npu_fusion_attention_v3.out": [
                    ("kwarg", "actual_seq_qlen", "actual_seq_qlen"),
                ],
            }

        class Record:
            class Op:
                __name__ = "npu_fusion_attention_v3.out"

            op_cache_entry = Op()
            kwargs = {}

        _NPU_GRAPH_OP_HANDLERS["npu_fusion_attention_v3.out"] = Handler
        validate_aclgraph_update_plan(
            [{"op": "npu_fusion_attention_v3.default", "updates": {
                "actual_seq_qlen": {"kind": "constant", "value": 4},
            }}],
            [Record()],
        )

    def test_validate_plan_rejects_empty_updates(self):
        class Record:
            class Op:
                __name__ = "npu_fusion_attention_v3.out"

            op_cache_entry = Op()
            kwargs = {}

        with self.assertRaisesRegex(RuntimeError, "no updates"):
            validate_aclgraph_update_plan(
                [{"op": "npu_fusion_attention_v3.out", "updates": {}}],
                [Record()],
            )

    def test_validate_plan_rejects_unsupported_constant(self):
        class Record:
            class Op:
                __name__ = "npu_fusion_attention_v3.out"

            op_cache_entry = Op()
            kwargs = {"actual_seq_qlen": None}

        with self.assertRaisesRegex(RuntimeError, "unsupported constant"):
            validate_aclgraph_update_plan(
                [{"op": "npu_fusion_attention_v3.out", "updates": {
                    "actual_seq_qlen": {"kind": "constant", "value": object()},
                }}],
                [Record()],
            )

    def test_validate_plan_rejects_unsupported_nested_constant(self):
        class Record:
            class Op:
                __name__ = "npu_fused_infer_attention_score.default"

            op_cache_entry = Op()
            kwargs = {"actual_seq_lengths": None}

        with self.assertRaisesRegex(RuntimeError, "unsupported constant"):
            validate_aclgraph_update_plan(
                [{"op": "npu_fused_infer_attention_score.default", "updates": {
                    "actual_seq_lengths": {"kind": "list", "items": [
                        {"kind": "constant", "value": object()},
                    ]},
                }}],
                [Record()],
            )

    def test_build_cpu_update_input_for_graph_tree(self):
        class Record:
            class Op:
                __name__ = "npu_fusion_attention_v3.out"

            op_cache_entry = Op()
            kwargs = {
                "actual_seq_qlen": None,
                "actual_seq_kvlen": None,
            }

        plan = [
            {
                "op": "npu_fusion_attention_v3.out",
                "updates": {
                    "actual_seq_qlen": {"kind": "input", "index": 0},
                    "actual_seq_kvlen": {"kind": "input", "index": 1},
                },
            }
        ]

        self.assertEqual(
            build_cpu_update_input_for_graph(plan, ["qlen", "kvlen"], [Record()]),
            [{"actual_seq_qlen": "qlen", "actual_seq_kvlen": "kvlen"}],
        )

if __name__ == "__main__":
    unittest.main()
