from unittest.mock import patch
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser import BasicDbParser


class TestBasicDbParser(TestCase):

    def setUp(self):
        self.test_dir = "temp"
        self.param_dict = {
            "profiler_path": self.test_dir,
            "output_path": self.test_dir
        }

    @patch('torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path',
           return_value=None)
    def test_basic_db_parser_run_no_cann_db(self, mock_get_cann_path):
        parser = BasicDbParser("test_basic_db", self.param_dict)

        with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.TorchDb') as mock_db:
            with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.ProfilerLogger'):
                mock_db_instance = mock_db.return_value
                mock_db_instance.create_connect_db.return_value = True

                mock_db_instance.judge_table_exist.return_value = False
                mock_db_instance.create_table_with_headers.return_value = None
                mock_db_instance.insert_data_into_table.return_value = None

                with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.ProfilerConfig') as mock_config:
                    mock_config_instance = mock_config.return_value
                    mock_config_instance.rank_id = 0

                    with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.ProfilerPathManager.get_device_id') as mock_device_id:
                        mock_device_id.return_value = [0]

                        with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.get_host_info') as mock_host_info:
                            mock_host_info.return_value = {"host_uid": "uid1", "host_name": "host1"}

                            with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.collect_env_vars') as mock_env_vars:
                                mock_env_vars.return_value = {"ENV_VARIABLES": {"key1": "value1"}}

                                with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.os.path.exists') as mock_exists:
                                    mock_exists.return_value = False
                                    status, result = parser.run({})
                                    self.assertEqual(status, Constant.SUCCESS)
                                    self.assertEqual(result, "")

    @patch('torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path',
           return_value="/mock/cann/path")
    def test_get_cann_db_path_no_valid_files(self, mock_get_cann_path):
        parser = BasicDbParser("test_basic_db", self.param_dict)
        with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.os.listdir') as mock_listdir:
            with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.os.path.exists') as mock_exists:
                with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.os.path.join') as mock_join:
                    with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.re.match') as mock_match:
                        with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.FileManager.check_db_file_vaild') as mock_check:
                            with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.ProfilerLogger'):
                                mock_listdir.return_value = ["invalid_file.txt"]
                                mock_exists.return_value = True

                                def mock_join_side_effect(x, y):
                                    return f"{x}/{y}"

                                mock_join.side_effect = mock_join_side_effect
                                mock_match.return_value = None
                                mock_check.side_effect = RuntimeError("Invalid file")
                                result = parser.get_cann_db_path()
                                self.assertEqual(result, "")

    @patch('torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path',
           return_value="/mock/cann/path")
    def test_save_profiler_metadata_to_db_json_error(self, mock_get_cann_path):
        parser = BasicDbParser("test_basic_db", self.param_dict)

        with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.os.path.exists') as mock_exists:
            with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.FileManager.file_read_all') as mock_read:
                with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.ProfilerLogger'):
                    mock_exists.return_value = True
                    mock_read.return_value = '{"invalid": json}'
                    parser.save_profiler_metadata_to_db()

    @patch('torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path',
           return_value="/mock/cann/path")
    def test_save_rank_info_to_db_multiple_devices(self, mock_get_cann_path):
        parser = BasicDbParser("test_basic_db", self.param_dict)
        with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.ProfilerConfig') as mock_config:
            with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.ProfilerLogger'):
                mock_config_instance = mock_config.return_value
                mock_config_instance.rank_id = 0

                with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.ProfilerPathManager.get_device_id') as mock_device_id:
                    mock_device_id.return_value = [0, 1]

                    with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.TorchDb') as mock_db:
                        mock_db_instance = mock_db.return_value
                        mock_db_instance.create_table_with_headers.return_value = None
                        mock_db_instance.insert_data_info_table.return_value = None

                        parser.save_rank_info_to_db()

    @patch('torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path',
           return_value="/mock/cann/path")
    def test_get_cann_db_path_from_mindstudio_output(self, mock_get_cann_path):
        parser = BasicDbParser("test_basic_db", self.param_dict)

        with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.os.listdir') as mock_listdir:
            with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.os.path.exists') as mock_exists:
                with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.os.path.join') as mock_join:
                    with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.re.match') as mock_match:
                        with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.FileManager.check_db_file_vaild') as mock_check:
                            with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.ProfilerLogger'):
                                mock_listdir.side_effect = [["invalid_file.txt"], ["msprof_123.db"]]
                                mock_exists.return_value = True

                                def mock_join_side_effect(x, y):
                                    return f"{x}/{y}"

                                mock_join.side_effect = mock_join_side_effect
                                mock_match.side_effect = [None, True]
                                mock_check.return_value = None

                                result = parser.get_cann_db_path()
                                self.assertEqual(result, "/mock/cann/path/mindstudio_profiler_output/msprof_123.db")

    @patch('torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path',
           return_value="/mock/cann/path")
    def test_basic_db_parser_run_db_connection_failure(self, mock_get_cann_path):
        parser = BasicDbParser("test_basic_db", self.param_dict)

        with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.TorchDb') as mock_db:
            with patch('torch_npu.profiler.analysis.prof_view.prof_db_parse._basic_db_parser.ProfilerLogger'):
                mock_db_instance = mock_db.return_value
                mock_db_instance.create_connect_db.return_value = False

                status, result = parser.run({})
                self.assertEqual(status, Constant.FAIL)
                self.assertEqual(result, "")

if __name__ == '__main__':
    run_tests()