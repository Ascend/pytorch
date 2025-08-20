from unittest.mock import patch

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.profiler.analysis.prof_view.prepare_parse._fwk_pre_parser import (
    TaskQueueParser, TorchOpParser, DbPreParser
)
from torch_npu.profiler.analysis.prof_common_func._constant import Constant


class TestFwkPreParsers(TestCase):

    def setUp(self):
        self.test_dir = "temp"
        self.param_dict = {
            "profiler_path": self.test_dir,
            "output_path": self.test_dir
        }

    @patch('torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path', return_value="/mock/cann/path")
    def test_task_queue_parser_should_return_success_with_enqueue_dequeue_data_when_run_with_valid_params(self, mock_get_cann_path):
        parser = TaskQueueParser("test_task_queue", self.param_dict)

        mock_enqueue_data = [{"name": "enqueue_op1", "ts": 1000}]
        mock_dequeue_data = [{"name": "dequeue_op1", "ts": 2000}]

        with patch('torch_npu.profiler.analysis.prof_view.prepare_parse._fwk_pre_parser.ProfilerLogger'):
            with patch('torch_npu.profiler.analysis.prof_view.prepare_parse._fwk_pre_parser.FwkFileParser') as mock_parser:
                mock_instance = mock_parser.return_value
                mock_instance.get_task_queue_data.return_value = (mock_enqueue_data, mock_dequeue_data)

                status, result = parser.run({})

                self.assertEqual(status, Constant.SUCCESS)
                self.assertEqual(result["enqueue_data"], mock_enqueue_data)
                self.assertEqual(result["dequeue_data"], mock_dequeue_data)

    @patch('torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path', return_value="/mock/cann/path")
    def test_torch_op_parser_should_return_success_with_torch_op_data_when_run_with_valid_params(self, mock_get_cann_path):
        parser = TorchOpParser("test_torch_op", self.param_dict)

        mock_torch_op_data = [{"name": "torch_op1", "ts": 1000}]

        with patch('torch_npu.profiler.analysis.prof_view.prepare_parse._fwk_pre_parser.ProfilerLogger'):
            with patch('torch_npu.profiler.analysis.prof_view.prepare_parse._fwk_pre_parser.FwkFileParser') as mock_parser:
                mock_instance = mock_parser.return_value
                mock_instance.get_file_data_by_tag.return_value = mock_torch_op_data

                status, result = parser.run({})

                self.assertEqual(status, Constant.SUCCESS)
                self.assertEqual(result, mock_torch_op_data)

    @patch('torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path', return_value="/mock/cann/path")
    def test_db_pre_parser_should_return_success_with_fwk_api_data_when_run_with_deps_data(self, mock_get_cann_path):
        parser = DbPreParser("test_db_pre", self.param_dict)

        mock_fwk_db_data = {"api_data": [{"name": "api1"}]}
        deps_data = {
            Constant.TORCH_OP_PARSER: [{"name": "torch_op1"}],
            Constant.TASK_QUEUE_PARSER: {
                "enqueue_data": [{"name": "enqueue_op1"}],
                "dequeue_data": [{"name": "dequeue_op1"}]
            }
        }

        with patch('torch_npu.profiler.analysis.prof_view.prepare_parse._fwk_pre_parser.ProfilerLogger'):
            with patch('torch_npu.profiler.analysis.prof_view.prepare_parse._fwk_pre_parser.FwkFileParser') as mock_parser:
                mock_instance = mock_parser.return_value
                mock_instance.get_fwk_api.return_value = mock_fwk_db_data

                status, result = parser.run(deps_data)

                self.assertEqual(status, Constant.SUCCESS)
                self.assertEqual(result, mock_fwk_db_data)


if __name__ == "__main__":
    run_tests()
