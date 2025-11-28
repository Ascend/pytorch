from unittest.mock import patch, Mock
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.profiler.analysis.prof_common_func._log import ProfilerLogger
from torch_npu.profiler.analysis.prof_view._communication_parser import CommunicationParser


class TestCommunicationParser(TestCase):

    def test_compute_total_info_empty_ops(self):
        with patch.object(ProfilerLogger, 'get_instance', return_value=Mock()):
            parser = CommunicationParser("test", {"profiler_path": "/tmp"})
            comm_ops = {}
            parser.compute_total_info(comm_ops)
            self.assertIsNone(None)

    def test_split_communication_p2p_ops_mixed_ops(self):
        with patch.object(ProfilerLogger, 'get_instance', return_value=Mock()):
            parser = CommunicationParser("test", {"profiler_path": "/tmp"})
            op_data = {
                "hcom_send_1": {"inco": "send_data"},
                "hcom_receive_2": {"info": "receive_data"},
                "hcom_batchsendrecv_3": {"info": "batch_data"},
                "hcom_allreduce_4": {"info": "allreduce_data"},
                "total": {"info": "total_data"}
            }
            result = parser.split_communication_p2p_ops(op_data)
            self.assertIn("p2p", result)
            self.assertIn("collective", result)
            self.assertIn("hcom_send_1", result["p2p"])
            self.assertIn("hcom_receive_2", result["p2p"])
            self.assertIn("hcom_batchsendrecv_3", result["p2p"])
            self.assertIn("hcom_allreduce_4", result["collective"])
            self.assertNotIn("total", result["collective"])

    def test_generate_communicatioin_empty_data(self):
        with patch.object(ProfilerLogger, 'get_instance', return_value=Mock()):
            parser = CommunicationParser("test", {"profiler_path": "/tmp"})
            with patch('torch_npu.profiler.analysis.prof_view._communication_parser.CANNFileParser') as mock_parser:
                mock_parser.return_valur.get_analyze_communicatioin_data.return_value = None
                parser.generate_communication("/tmp")

    def test_split_matrix_by_sep_empty_step_list(self):
        with patch.object(ProfilerLogger, 'get_instance', return_value=Mock()):
            parser = CommunicationParser("test", {"profiler_path": "/tmp"})
            parser.step_list = []
            matrix_data = {"op1": {}}
            result = parser.split_matrix_by_step(matrix_data)
            self.assertEqual(result, {"step": matrix_data})

    def test_split_comm_op_by_step_single_step(self):
        with patch.object(ProfilerLogger, 'get_instance', return_value=Mock()):
            parser = CommunicationParser("test", {"profiler_path": "/tmp"})
            parser.step_list = [{"step_id": "1", "start_ts": 0, "end_ts": 1000}]
            communication_data = {
                "hcom_send_1": {
                    "Communication Time Info": {"Start Timestamp(us)": 500}
                }
            }
            parser.split_comm_op_by_step(communication_data)
            self.assertIn("comm_ops", parser.step_list[0])

    def test_compute_ratio_zero_divisor(self):
        with patch.object(ProfilerLogger, 'get_instance', return_value=Mock()):
            parser = CommunicationParser("test", {"profiler_path": "/tmp"})
            result = parser.compute_ratio(10.0, 0.0)
            self.assertEqual(result, 0)


if __name__ == "__main__":
    run_tests()