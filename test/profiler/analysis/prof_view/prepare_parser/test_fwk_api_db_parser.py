from unittest.mock import patch
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.analysis.prof_view.prof_db_parse._fwk_api_db_parser import FwkApiDbParser


class TestFwkApiDbParser(TestCase):

    def setUp(self):
        self.test_dir = "temp"
        self.param_dict = {
            "profiler_path": self.test_dir,
            "output_patch": self.test_dir
        }

    @patch('torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path',
           return_value="/mock/cann/path")
    def test_fwk_api_db_parser_run_db_connection_failure(self, mock_get_cann_path):
        parser = FwkApiDbParser("test_fwk_api", self.param_dict)
        deps_data = {Constant.DB_PRE_PARSER: {}}
        with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._fwk_api_db_parser.TorchDb') as mock_db:
            mock_db_instance = mock_db.return_value
            mock_db_instance.create_connect_db.return_value = False
            status, result = parser.run(deps_data)
            self.assertEqual(status, Constant.FAIL)
            self.assertIsNone(result)

    @patch('torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path',
           return_value="/mock/cann/path")
    def test_get_api_data_for_db_empty_input(self, mock_get_cann_path):
        parser = FwkApiDbParser("test_fwk_api", self.param_dict)
        fwk_api_data = {}
        parser.get_api_data_for_db(fwk_api_data)
        self.assertEqual(parser._fwk_apis, [])

    def test_get_api_data_for_db_empty_data_types(self):
        parser = FwkApiDbParser("test_fwk_api", self.param_dict)
        fwk_api_data = {
            Constant.ENQUEUE_DATA: [],
            Constant.DEQUEUE_DATA: [],
            Constant.TORCH_OP_DATA: [],
            Constant.PYTHON_TRACE_DATA: [],
            Constant.MSTX_OP_DATA: []
        }
        with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._fwk_api_db_parser.TorchDb') as mock_db:
            mock_db_instance = mock_db.return_value
            mock_db_instance.create_connect_db.return_value = True
            mock_db_instance.judge_table_exist.return_value = False
            parser.get_api_data_for_db(fwk_api_data)
            self.assertEqual(parser._fwk_apis, [])

    def test_get_torch_op_connection_ids_with_task_queue_empty_queues(self):
        parser = FwkApiDbParser("test_fwk_api", self.param_dict)
        task_enqueues = []
        task_dequeues = []
        torch_op_apis = [{"name": "torch_op1", "ts": 1500, "connection_id": []}]
        node_launch_apis = [{"startNs": 1000, "endNs": 2000, "globalTid": 1, "correlationId": 1}]
        with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._fwk_api_db_parser.ConnectionIdManager') as mock_conn_manager:
            mock_conn_manager_instance = mock_conn_manager.return_value
            mock_conn_manager_instance.get_connection_ids_from_id.return_value = [1]
            parser.get_torch_op_connection_ids_with_task_queue(task_enqueues, task_dequeues, torch_op_apis, len(torch_op_apis), node_launch_apis)

    def test_get_mstx_mark_op_connection_ids_with_cann_api_no_cann_tx_apis(self):
        parser = FwkApiDbParser("test_fwk_api", self.param_dict)
        task_enqueues = []
        task_dequeues = []
        torch_op_apis = [{"name": "mstx_op1", "ts": 1300}]
        with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._fwk_api_db_parser.TorchDb') as mock_db:
            mock_db_instance = mock_db.return_value
            mock_db_instance.fetch_all_data.return_value = []
            with self.assertRaises(RuntimeWarning) as context:
                parser.get_mstx_mark_op_connection_ids_with_cann_api(task_enqueues, task_dequeues, torch_op_apis)
            self.assertIn("Failed to get msprof_tx apis", str(context.exception))

    def test_save_api_data_to_db_calls_all_save_methods(self):
        parser = FwkApiDbParser("test_fwk_api", self.param_dict)
        parser._fwk_apis = [{"name": "test_api"}]
        with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._fwk_api_db_parser.TorchDb') as mock_db:
            mock_db_instance = mock_db.return_value
            mock_db_instance.judge_table_exist.return_value = True
            with patch.object(parser, 'save_fwk_api') as mock_save_fwk_api:
                with patch.object(parser, 'save_string_ids') as mock_save_string_ids:
                    with patch.object(parser, 'sava_connection_ids') as mock_save_connection_ids:
                        with patch.object(parser, 'save_callchain_ids') as mock_save_callchain_ids:
                            with patch.object(parser, 'save_enum_api_types_to_db') as mock_save_enum_api_types:
                                parser.save_api_data_to_db()
                                mock_save_fwk_api.assert_called_once()
                                mock_save_string_ids.assert_called_once()
                                mock_save_connection_ids.assert_called_once()
                                mock_save_callchain_ids.assert_called_once()
                                mock_save_enum_api_types.assert_called_once()

    def test_get_torch_op_connection_ids_with_cann_api_empty_apis(self):
        parser = FwkApiDbParser("test_fwk_api", self.param_dict)
        task_enqueues = []
        task_dequeues = []
        torch_op_apis = []
        with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._fwk_api_db_parser.TorchDb') as mock_db:
            mock_db_instance = mock_db.return_value
            mock_db_instance.fetch_one_data.return_value = [1]
            mock_db_instance.fetch_all_data.return_value = []
            parser.get_torch_op_connection_ids_with_cann_api(task_enqueues, task_dequeues, torch_op_apis)

    def test_get_mstx_mark_op_connection_ids_with_cann_api_empty_apis(self):
        parser = FwkApiDbParser("test_fwk_api", self.param_dict)
        task_enqueues = []
        task_dequeues = []
        mstx_mark_apis = []
        with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._fwk_api_db_parser.TorchDb') as mock_db:
            mock_db_instance = mock_db.return_value
            mock_db_instance.fetch_all_data.return_value = []
            parser.get_mstx_mark_op_connection_ids_with_cann_api(task_enqueues, task_dequeues, mstx_mark_apis)

if __name__ == '__main__':
    run_tests()