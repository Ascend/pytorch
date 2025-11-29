from unittest.mock import patch, MagicMock
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.analysis.prof_parse._fwk_cann_relation_parser import FwkCANNRelationParser


class TestFwkFileParser(TestCase):

    def test_merge_disjoint(self):
        acl_to_npu_dict = {1000000: ["kernel1"], 2000000: ["kernel2"]}
        dequeue_data_list = [
            MagicMock(ts=500000, dur=100000, corr_id=10),
            MagicMock(ts=1500000, dur=100000, corr_id=20)
        ]

        result = FwkCANNRelationParser.combine_kernel_dict(acl_to_npu_dict, dequeue_data_list)

        expected = {}
        self.assertEqual(result, expected)


    def test_get_step_range_empty_step_node_list(self):
        root_node = MagicMock()
        root_node.child_node_list = []

        parser = FwkCANNRelationParser("test_path")
        result = parser.get_step_range(root_node, {"1000000": ["kernel1"]})
        self.assertEqual(result, [])

    def test_get_kernel_dict_empty_acl_to_npu_with_none_level(self):
        with patch('torch_npu.profiler.analysis.prof_parse._fwk_cann_relation_parser.CANNFileParser') as mock_parser, \
                patch('torch_npu.profiler.analysis.prof_parse._fwk_cann_relation_parser.ProfilerConfig') as mock_config:
            mock_parser.return_value.get_acl_to_npu_data.return_value = {}
            mock_config.return_value.get_npu_level.return_value = Constant.LEVEL_NONE

            parser = FwkCANNRelationParser("test_path")
            result = parser.get_kernel_dict([])
            self.assertEqual(result, {})

    def test_get_step_range_empty_kernel_dict(self):
        mock_root_node = MagicMock()
        mock_root_node.child_node_list = []
        parser = FwkCANNRelationParser("test_path")
        result = parser.get_step_range(mock_root_node, [])
        self.assertEqual(result, [])

    def test_combine_kernel_dict_empty_dequeue_list(self):
        acl_to_npu_dict = {1000000: ["kernel1", "kernel2"]}
        dequeue_data_list = []
        result = FwkCANNRelationParser.combine_kernel_dict(acl_to_npu_dict, dequeue_data_list)
        self.assertEqual(result, acl_to_npu_dict)

    def test_update_nodes_overlap(self):
        step_node_list = [
            MagicMock(start_time=1000000, end_time=2000000, corr_id_total=None),
            MagicMock(start_time=2500000, end_time=3500000, corr_id_total=None),
        ]
        acl_start_time_list = [1500000, 3000000]

        step_node_list[0].update_corr_id_total = MagicMock()
        step_node_list[1].update_corr_id_total = MagicMock()

        FwkCANNRelationParser._update_step_node_info(step_node_list, acl_start_time_list)

        step_node_list[0].update_corr_id_total.assert_called_once_with(1500000)
        step_node_list[1].update_corr_id_total.assert_called_once_with(3000000)


if __name__ == "__main__":
    run_tests()