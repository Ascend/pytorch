import unittest
from unittest.mock import MagicMock

from torch_npu.profiler.analysis.prof_bean._torch_op_node import TorchOpNode
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.testing.testcase import TestCase, run_tests


class TestTorchOpNode(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # root event
        root_event = MagicMock()
        root_event.pid = 999
        root_event.name = "ProfilerStep#1"
        root_event.args = {Constant.INPUT_SHAPES: "[2, 2048]"}
        root_event.ts = 10
        root_event.end_ns = 100
        root_event.dur = 90
        root_event.call_stack = "call stack string0"
        root_node = TorchOpNode(root_event)
        # level1 event1
        level1_event1 = MagicMock()
        level1_event1.pid = 999
        level1_event1.name = "MatMul"
        level1_event1.args = {Constant.INPUT_SHAPES: "[2, 2048]"}
        level1_event1.ts = 20
        level1_event1.end_ns = 30
        level1_event1.dur = 10
        level1_event1.call_stack = "call stack string1"
        level1_node1 = TorchOpNode(level1_event1, root_node)
        root_node.add_child_node(level1_node1)
        # level1 event2
        level1_event2 = MagicMock()
        level1_event2.pid = 999
        level1_event2.name = "Div"
        level1_event2.args = {Constant.INPUT_SHAPES: "[2, 2048]"}
        level1_event2.ts = 40
        level1_event2.end_ns = 80
        level1_event2.dur = 40
        level1_event2.call_stack = "call stack string2"
        level1_node2 = TorchOpNode(level1_event2, root_node)
        root_node.add_child_node(level1_node2)
        # level2 event
        level2_event = MagicMock()
        level2_event.pid = 999
        level2_event.name = "Add"
        level2_event.args = {Constant.INPUT_SHAPES: "[2, 2048]"}
        level2_event.ts = 60
        level2_event.end_ns = 70
        level2_event.dur = 10
        level2_event.call_stack = "call stack string3"
        level2_node = TorchOpNode(level2_event, level1_node2)
        level1_node2.add_child_node(level2_node)

        cls.root_node = root_node
        cls.level1_node1 = level1_node1
        cls.level1_node2 = level1_node2
        cls.level2_node = level2_node

    def test_property(self):
        # simple property: only test the root
        self.assertEqual(self.root_node.pid, 999)
        self.assertEqual(self.root_node.name, "ProfilerStep#1")
        self.assertEqual(self.root_node.is_profiler_step(), True)
        self.assertEqual(self.root_node.input_shape, "[2, 2048]")
        self.assertEqual(self.root_node.call_stack, "call stack string0")
        self.assertEqual(self.root_node.start_time, 10)
        self.assertEqual(self.root_node.end_time, 100)
        # complex property
        self.assertEqual(self.root_node.host_self_dur, 40)
        self.assertEqual(self.root_node.host_total_dur, 90)
        self.assertEqual(self.root_node.child_node_list, [self.level1_node1, self.level1_node2])
        self.assertEqual(self.root_node.parent_node, None)
        self.assertEqual(self.level1_node1.host_self_dur, 10)
        self.assertEqual(self.level1_node2.host_self_dur, 30)
        self.assertEqual(self.level1_node2.host_total_dur, 40)
        self.assertEqual(self.level1_node2.child_node_list, [self.level2_node])

    @unittest.skip("skip test_match_child_node now")
    def test_match_child_node(self):
        self.assertEqual(self.root_node.match_child_node(0), None)
        self.assertEqual(self.root_node.match_child_node(35), None)
        self.assertEqual(self.root_node.match_child_node(25), self.level1_node1)
        self.assertEqual(self.root_node.match_child_node(50), self.level1_node2)
        self.assertEqual(self.level1_node2.match_child_node(65), self.level2_node)

    def test_updata_corr_id(self):
        self.root_node.update_corr_id(10)
        self.level1_node1.update_corr_id(20)
        self.level1_node2.update_corr_id(30)
        self.level2_node.update_corr_id(40)
        self.assertEqual(self.root_node.corr_id_self, [10])
        self.assertEqual(self.root_node.corr_id_total, [])
        self.assertEqual(self.level1_node1.corr_id_self, [20])
        self.assertEqual(self.level1_node1.corr_id_total, [20])
        self.assertEqual(self.level1_node2.corr_id_self, [30])
        self.assertEqual(self.level1_node2.corr_id_total, [30, 40])
        self.assertEqual(self.level2_node.corr_id_self, [40])
        self.assertEqual(self.level2_node.corr_id_total, [40])


if __name__ == "__main__":
    run_tests()
