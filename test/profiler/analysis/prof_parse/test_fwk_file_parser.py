import os
import tempfile
from unittest.mock import patch, MagicMock

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.profiler.analysis.prof_parse._fwk_file_parser import FwkFileParser
from torch_npu.profiler.analysis.prof_common_func._constant import Constant, ApiType


class MockTorchOpBean:
    """Mock TorchOpBean for testing"""
    def __init__(self, pid=12345, tid=67890, name="test_op", ts=1000000, end_ns=2000000, args=None):
        self.pid = pid
        self.tid = tid
        self.name = name
        self.ts = ts
        self.end_ns = end_ns
        self.dur = end_ns - ts
        self.args = args or {}


class MockOpMarkBean:
    """Mock OpMarkBean for testing"""
    def __init__(self, pid=12345, tid=67890, name="test_mark", ts=1000000, dur=500000,
                 corr_id=1, origin_name="origin_test", category=1):
        self.pid = pid
        self.tid = tid
        self.name = name
        self.ts = ts
        self.time_ns = ts
        self.dur = dur
        self.corr_id = corr_id
        self.origin_name = origin_name
        self.args = {"correlation_id": corr_id}
        self._category = category


class TestFwkFileParser(TestCase):

    def setUp(self):
        self.test_dir = "temp"
        self.fwk_dir = os.path.join(self.test_dir, "FRAMEWORK")

    @patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.ProfilerLogger')
    @patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.ProfilerPathManager')
    @patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.os.listdir')
    @patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.os.path.isfile')
    def test_get_fwk_trace_data_should_return_valid_list_when_given_torch_ops_and_taskqueue(self, mock_isfile, mock_listdir, mock_path_manager, mock_logger):
        """Test basic trace data generation"""
        mock_path_manager.get_fwk_path.return_value = self.fwk_dir
        mock_listdir.return_value = []
        mock_isfile.return_value = True
        mock_logger.init.return_value = None
        mock_logger.get_instance.return_value = MagicMock()

        torch_op_data = [
            MockTorchOpBean(pid=12345, tid=1001, name="conv2d", ts=1000000, end_ns=2000000),
            MockTorchOpBean(pid=12345, tid=1001, name="matmul", ts=2000000, end_ns=3000000)
        ]
        enqueue_data = [
            MockOpMarkBean(pid=12345, tid=1001, name="Enqueue@conv2d", ts=1000000, dur=1000000, corr_id=10),
            MockOpMarkBean(pid=12345, tid=1001, name="Enqueue@matmul", ts=2000000, dur=1000000, corr_id=20)
        ]
        dequeue_data = [
            MockOpMarkBean(pid=12345, tid=1002, name="Dequeue@conv2d", ts=2000000, dur=1000000, corr_id=10),
            MockOpMarkBean(pid=12345, tid=1002, name="Dequeue@matmul", ts=3000000, dur=1000000, corr_id=20)
        ]

        parser = FwkFileParser(self.test_dir)

        with patch.object(parser, 'get_python_trace_data', return_value=[]), \
             patch.object(parser, 'get_gc_record_trace_data', return_value=[]):

            result = parser.get_fwk_trace_data(torch_op_data, enqueue_data, dequeue_data)

            # Verify results
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 17)  # 6 x event + 2 e event + 2 f event + 7 m event

    @patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.ProfilerLogger')
    @patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.ProfilerPathManager')
    @patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.os.listdir')
    @patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.os.path.isfile')
    def test_get_fwk_api_should_return_valid_dict_when_given_torch_ops(self, mock_isfile, mock_listdir, mock_path_manager, mock_logger):
        """Test basic API data generation"""
        # Setup mocks
        mock_path_manager.get_fwk_path.return_value = self.fwk_dir
        mock_listdir.return_value = []
        mock_isfile.return_value = True
        mock_logger.init.return_value = None
        mock_logger.get_instance.return_value = MagicMock()

        torch_op_data = [
            MockTorchOpBean(pid=12345, tid=1001, name="conv2d", ts=1000000, end_ns=2000000,
                          args={Constant.SEQUENCE_NUMBER: 100, Constant.FORWARD_THREAD_ID: 0})
        ]

        parser = FwkFileParser(self.test_dir)

        with patch.object(parser, 'get_file_data_by_tag', return_value=[]), \
             patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.Str2IdManager') as mock_str_mgr, \
             patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.ConnectionIdManager') as mock_conn_mgr, \
             patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.CallChainIdManager') as mock_chain_mgr:

            mock_str_instance = MagicMock()
            mock_str_instance.get_id_from_str.return_value = 1001
            mock_str_mgr.return_value = mock_str_instance

            mock_conn_instance = MagicMock()
            mock_conn_mgr.return_value = mock_conn_instance

            mock_chain_instance = MagicMock()
            mock_chain_mgr.return_value = mock_chain_instance

            result = parser.get_fwk_api(torch_op_data, [], [])

            # Verify results
            self.assertIsInstance(result, dict)
            self.assertIn(Constant.TORCH_OP_DATA, result)
            self.assertEqual(len(result[Constant.TORCH_OP_DATA]), 1)
            api_data = result[Constant.TORCH_OP_DATA][0]
            self.assertEqual(api_data[0], 1000000)
            self.assertEqual(api_data[1], 2000000)
            self.assertEqual(api_data[10], ApiType.TORCH_OP)
            self.assertEqual(len(api_data), 11)

    @patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.ProfilerLogger')
    @patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.ProfilerPathManager')
    @patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.os.listdir')
    @patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.os.path.isfile')
    def test_get_fwk_api_should_process_task_queue_data_when_given_enqueue_dequeue_ops(self, mock_isfile, mock_listdir, mock_path_manager, mock_logger):
        """Test API data generation with task queue"""
        # Setup mocks
        mock_path_manager.get_fwk_path.return_value = self.fwk_dir
        mock_listdir.return_value = []
        mock_isfile.return_value = True
        mock_logger.init.return_value = None
        mock_logger.get_instance.return_value = MagicMock()

        enqueue_data = [MockOpMarkBean(name="enqueue_op", ts=500000, dur=100000, corr_id=1)]
        dequeue_data = [MockOpMarkBean(name="dequeue_op", ts=1500000, dur=200000, corr_id=1)]

        parser = FwkFileParser(self.test_dir)

        with patch.object(parser, 'get_file_data_by_tag', return_value=[]), \
             patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.Str2IdManager') as mock_str_mgr, \
             patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.ConnectionIdManager') as mock_conn_mgr, \
             patch('torch_npu.profiler.analysis.prof_parse._fwk_file_parser.CallChainIdManager') as mock_chain_mgr:

            mock_str_instance = MagicMock()
            mock_str_instance.get_id_from_str.return_value = 2001
            mock_str_mgr.return_value = mock_str_instance

            mock_conn_instance = MagicMock()
            mock_conn_instance.get_id_from_connection_ids.return_value = 1
            mock_conn_mgr.return_value = mock_conn_instance

            mock_chain_instance = MagicMock()
            mock_chain_mgr.return_value = mock_chain_instance

            result = parser.get_fwk_api([], enqueue_data, dequeue_data)

            # Verify results
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result[Constant.ENQUEUE_DATA]), 1)
            self.assertEqual(len(result[Constant.DEQUEUE_DATA]), 1)
            enqueue = result[Constant.ENQUEUE_DATA][0]
            dequeue = result[Constant.DEQUEUE_DATA][0]
            self.assertEqual(enqueue[0], 500000)    # enqueue start_ns
            self.assertEqual(dequeue[0], 1500000)   # dequeue start_ns
            self.assertEqual(enqueue[10], ApiType.TASK_QUEUE)  # api_type
            self.assertEqual(dequeue[10], ApiType.TASK_QUEUE)  # api_type
            self.assertEqual(len(enqueue), 11)
            self.assertEqual(len(dequeue), 11)


if __name__ == "__main__":
    run_tests()
