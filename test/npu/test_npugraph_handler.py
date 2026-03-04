from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.npu import (
    NpuGraphOpHandler,
    register_npu_graph_handler,
)
from torch_npu.npu._npugraph_handlers.npugraph_handler import _NPU_GRAPH_OP_HANDLERS


class TestNpuGraphHandlerRegistry(TestCase):

    def setUp(self):
        self._snapshot = dict(_NPU_GRAPH_OP_HANDLERS)

    def tearDown(self):
        _NPU_GRAPH_OP_HANDLERS.clear()
        _NPU_GRAPH_OP_HANDLERS.update(self._snapshot)

    def test_register_single_name(self):
        @register_npu_graph_handler("test_op_single")
        class _H(NpuGraphOpHandler):
            pass

        self.assertIn("test_op_single", _NPU_GRAPH_OP_HANDLERS)
        self.assertIs(_NPU_GRAPH_OP_HANDLERS["test_op_single"], _H)

    def test_register_multiple_names(self):
        @register_npu_graph_handler(["test_op_a", "test_op_a.default"])
        class _H(NpuGraphOpHandler):
            pass

        self.assertIn("test_op_a", _NPU_GRAPH_OP_HANDLERS)
        self.assertIn("test_op_a.default", _NPU_GRAPH_OP_HANDLERS)
        self.assertIs(_NPU_GRAPH_OP_HANDLERS["test_op_a"], _H)


class TestNpuGraphHandlerBuiltinRegistration(TestCase):

    EXPECTED_OPS = [
        "npu_fused_infer_attention_score",
        "npu_fused_infer_attention_score.default",
        "npu_fused_infer_attention_score.out",
        "npu_fused_infer_attention_score_v2",
        "npu_fused_infer_attention_score_v2.default",
        "npu_fused_infer_attention_score_v2.out",
        "_npu_paged_attention.default",
        "npu_multi_head_latent_attention.out",
    ]

    def test_all_expected_ops_registered(self):
        for op_name in self.EXPECTED_OPS:
            with self.subTest(op_name=op_name):
                self.assertIn(
                    op_name,
                    _NPU_GRAPH_OP_HANDLERS,
                    f"Expected handler for '{op_name}' not found in registry",
                )


if __name__ == "__main__":
    run_tests()
