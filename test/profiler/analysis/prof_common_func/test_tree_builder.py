from unittest.mock import MagicMock

from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.analysis.prof_common_func._tree_builder import TreeBuilder
from torch_npu.testing.testcase import TestCase, run_tests


class TestTreeBuilder(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        level0_event = MagicMock()
        level0_event.pid = 999
        level0_event.name = "ProfilerStep#1"
        level0_event.args = {Constant.INPUT_SHAPES: "[2, 2048]", Constant.CALL_STACK: "call stack string0"}
        level0_event.ts = 10
        level0_event.end_ns = 100
        level0_event.dur = 90
        level0_event.is_torch_op = True
        # level1 event1
        level1_event1 = MagicMock()
        level1_event1.pid = 999
        level1_event1.name = "MatMul"
        level1_event1.args = {Constant.INPUT_SHAPES: "[2, 2048]", Constant.CALL_STACK: "call stack string1"}
        level1_event1.ts = 20
        level1_event1.end_ns = 30
        level1_event1.dur = 10
        level1_event1.is_torch_op = True
        # level1 event1
        level1_event2 = MagicMock()
        level1_event2.pid = 999
        level1_event2.name = "MatMul"
        level1_event2.args = {Constant.INPUT_SHAPES: "[2, 2048]", Constant.CALL_STACK: "call stack string1"}
        level1_event2.ts = 50
        level1_event2.end_ns = 80
        level1_event2.dur = 10
        level1_event2.is_torch_op = True
        # level2 event
        level2_event = MagicMock()
        level2_event.pid = 999
        level2_event.name = "Add"
        level2_event.args = {Constant.INPUT_SHAPES: "[2, 2048]", Constant.CALL_STACK: "call stack string3"}
        level2_event.ts = 60
        level2_event.end_ns = 70
        level2_event.dur = 10
        level2_event.is_torch_op = True

        cls.event_list = [level0_event, level1_event1, level1_event2, level2_event]

    def test_build_tree(self):
        nodes = TreeBuilder.build_tree(self.event_list, [])
        self.assertEqual(len(self.event_list) + 1, len(nodes))
        level0_node = nodes[1]
        level1_node2 = nodes[3]
        level2_node = nodes[4]
        self.assertEqual(2, len(level0_node.child_node_list))
        self.assertEqual(1, len(level1_node2.child_node_list))
        self.assertEqual(level2_node, level1_node2.child_node_list[0])

    def test_update_tree_node_info(self):
        nodes = TreeBuilder.build_tree(self.event_list, [])
        root_node = nodes[0]
        ts_list = [25, 40, 65]
        for ts in ts_list:
            TreeBuilder.update_tree_node_info(ts, root_node)
        self.assertEqual(nodes[1].corr_id_self, [40])
        self.assertEqual(nodes[1].corr_id_total, [25, 40, 65])
        self.assertEqual(nodes[2].corr_id_self, [25])
        self.assertEqual(nodes[4].corr_id_self, [65])

    def test_match_self_torch_op(self):
        nodes = TreeBuilder.build_tree(self.event_list, [])
        root_node = nodes[0]
        match_op = TreeBuilder.match_self_torch_op(25, root_node)
        self.assertEqual(match_op, nodes[2])
        match_op = TreeBuilder.match_self_torch_op(40, root_node)
        self.assertEqual(match_op, nodes[1])
        match_op = TreeBuilder.match_self_torch_op(65, root_node)
        self.assertEqual(match_op, nodes[4])


if __name__ == "__main__":
    run_tests()
