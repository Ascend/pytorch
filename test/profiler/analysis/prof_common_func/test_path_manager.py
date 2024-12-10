import os
import shutil
import stat
import json

from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.analysis.prof_common_func._file_manager import FileManager
from torch_npu.profiler.analysis.prof_common_func._path_manager import ProfilerPathManager

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.utils._path_manager import PathManager


class TestPathManager(TestCase):

    def setUp(self):
        self.tmp_dir = "./tmp_dir"
        os.makedirs(self.tmp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_get_fwk_path(self):
        self.assertEqual("", ProfilerPathManager.get_fwk_path(self.tmp_dir))
        fwk_path = os.path.join(self.tmp_dir, Constant.FRAMEWORK_DIR)
        os.makedirs(fwk_path)
        self.assertEqual(fwk_path, ProfilerPathManager.get_fwk_path(self.tmp_dir))

    def test_get_cann_path(self):
        self.assertEqual("", ProfilerPathManager.get_cann_path(self.tmp_dir))
        cann_path = os.path.realpath(os.path.join(self.tmp_dir, "PROF_test"))
        os.makedirs(cann_path)
        self.assertEqual("", ProfilerPathManager.get_cann_path(cann_path))
        cann_path = os.path.join(self.tmp_dir, "PROF_1_2_3a")
        os.makedirs(cann_path)
        self.assertEqual(cann_path, ProfilerPathManager.get_cann_path(self.tmp_dir))

    def test_get_info_file_path(self):
        self.assertEqual("", ProfilerPathManager.get_info_file_path(self.tmp_dir))
        info_file = os.path.join(self.tmp_dir, "profiler_info.json")
        FileManager.create_json_file_by_path(info_file, {"Name": "test"})
        self.assertEqual(info_file, ProfilerPathManager.get_info_file_path(self.tmp_dir))
        os.remove(info_file)
        info_file = os.path.join(self.tmp_dir, "profiler_info_2.json")
        FileManager.create_json_file_by_path(info_file, {"Name": "test"})
        self.assertEqual(info_file, ProfilerPathManager.get_info_file_path(self.tmp_dir))

    def test_get_info_path(self):
        self.assertEqual("", ProfilerPathManager.get_info_path(self.tmp_dir))
        os.makedirs(os.path.join(self.tmp_dir, "PROF_1_2_3a", "host"))
        info_path = os.path.join(self.tmp_dir, "PROF_1_2_3a", "host", "info.json")
        FileManager.create_json_file_by_path(info_path, {"Name": "Test"})
        self.assertEqual(info_path, ProfilerPathManager.get_info_path(self.tmp_dir))

    def test_host_start_log_path(self):
        self.assertEqual("", ProfilerPathManager.get_host_start_log_path(self.tmp_dir))
        os.makedirs(os.path.join(self.tmp_dir, "PROF_1_2_3a", "host"))
        log_path = os.path.join(self.tmp_dir, "PROF_1_2_3a", "host", "host_start.log")
        with os.fdopen(os.open(log_path,
                               os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w') as fp:
            fp.write("something")
        self.assertEqual(log_path, ProfilerPathManager.get_host_start_log_path(self.tmp_dir))

    def test_get_host_path(self):
        self.assertEqual("", ProfilerPathManager.get_host_path(self.tmp_dir))
        cann_path = os.path.join(self.tmp_dir, "PROF_1_2_3a")
        host_path = os.path.join(cann_path, "host")
        os.makedirs(host_path)
        self.assertEqual(host_path, ProfilerPathManager.get_host_path(cann_path))

    def test_get_device_path(self):
        self.assertEqual("", ProfilerPathManager.get_device_path(self.tmp_dir))
        cann_path = os.path.join(self.tmp_dir, "PROF_1_2_3a")
        device_path = os.path.join(cann_path, "device_0")
        os.makedirs(device_path)
        self.assertEqual(device_path, ProfilerPathManager.get_device_path(cann_path))

    def test_get_device_id(self):
        self.assertEqual(Constant.INVALID_VALUE, ProfilerPathManager.get_device_id(""))
        cann_path = os.path.join(self.tmp_dir, "PROF_1_2_3a")
        os.makedirs(cann_path)
        self.assertEqual(Constant.INVALID_VALUE, ProfilerPathManager.get_device_id(cann_path))
        invalid_device_path = os.path.join(cann_path, "device")
        os.makedirs(invalid_device_path)
        self.assertEqual(Constant.INVALID_VALUE, ProfilerPathManager.get_device_id(cann_path))
        invalid_device_path = os.path.join(cann_path, "device_xx")
        self.assertEqual(Constant.INVALID_VALUE, ProfilerPathManager.get_device_id(cann_path))
        invalid_device_path = os.path.join(cann_path, "device_0")
        os.makedirs(invalid_device_path)
        self.assertEqual(0, ProfilerPathManager.get_device_id(cann_path))

    def test_get_start_info_path(self):
        self.assertEqual("", ProfilerPathManager.get_start_info_path(self.tmp_dir))
        cann_path = os.path.join(self.tmp_dir, "PROF_1_2_3a")
        # search device directory
        device_path = os.path.join(cann_path, "device_1_3")
        os.makedirs(device_path)
        self.assertEqual("", ProfilerPathManager.get_start_info_path(cann_path))
        shutil.rmtree(device_path)
        device_path = os.path.join(cann_path, "device_1")
        os.makedirs(device_path)
        self.assertEqual("", ProfilerPathManager.get_start_info_path(cann_path))
        with os.fdopen(os.open(os.path.join(device_path, "start_info.1"),
                               os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w') as fp:
            fp.write("something")
        self.assertEqual(os.path.join(device_path, "start_info.1"), ProfilerPathManager.get_start_info_path(cann_path))
        # search host directory
        host_path = os.path.join(cann_path, "host")
        os.makedirs(host_path)
        with os.fdopen(os.open(os.path.join(host_path, "start_info"),
                               os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w') as fp:
            fp.write("something")
        self.assertEqual(os.path.join(host_path, "start_info"), ProfilerPathManager.get_start_info_path(cann_path))

    def test_get_profiler_path_list(self):
        self.assertEqual([], ProfilerPathManager.get_profiler_path_list("Somthing"))
        prof_path1 = os.path.join(self.tmp_dir, "test1")
        os.makedirs(os.path.join(prof_path1, Constant.FRAMEWORK_DIR))
        prof_path2 = os.path.join(self.tmp_dir, "test2")
        os.makedirs(os.path.join(prof_path2, "PROF_1_2_3a"))
        self.assertEqual([prof_path1], ProfilerPathManager.get_profiler_path_list(prof_path1))
        self.assertEqual([prof_path2], ProfilerPathManager.get_profiler_path_list(prof_path2))
        self.assertEqual(set((prof_path1, prof_path2)), set(ProfilerPathManager.get_profiler_path_list(self.tmp_dir)))

    def test_device_all_file_list_by_tag(self):
        self.assertEqual([], ProfilerPathManager.get_output_all_file_list_by_type(self.tmp_dir, "mindstudio_profiler_output"))
        cann_path = os.path.join(self.tmp_dir, "PROF_1_2_3a")
        output_path = os.path.join(cann_path, "mindstudio_profiler_output")
        os.makedirs(output_path)
        self.assertEqual([], ProfilerPathManager.get_output_all_file_list_by_type(cann_path, "mindstudio_profiler_output"))
        with os.fdopen(os.open(os.path.join(output_path, "test_file1.log"),
                               os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w') as fp:
            fp.write("something")
        with os.fdopen(os.open(os.path.join(output_path, "test_file2.log"),
                               os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w') as fp:
            fp.write("something")
        with os.fdopen(os.open(os.path.join(output_path, "test_file3.log"),
                               os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w') as fp:
            fp.write("something")
        self.assertEqual(3, len(ProfilerPathManager.get_output_all_file_list_by_type(cann_path, "mindstudio_profiler_output")))

    def test_get_feature_json_path(self):
        self.assertEqual("", ProfilerPathManager.get_feature_json_path(self.tmp_dir))

        cann_path = os.path.join(self.tmp_dir, "PROF_001_111111_AAA")
        PathManager.make_dir_safety(cann_path)
        self.assertEqual("", ProfilerPathManager.get_feature_json_path(self.tmp_dir))

        host_path = os.path.join(cann_path, "host")
        PathManager.make_dir_safety(host_path)
        self.assertEqual("", ProfilerPathManager.get_feature_json_path(self.tmp_dir))

        feature_file = os.path.join(host_path, "incompatible_features.json")
        FileManager.create_json_file_by_path(feature_file, {"attr": {"version": "1"}})
        self.assertEqual(feature_file, ProfilerPathManager.get_feature_json_path(self.tmp_dir))

    def test_get_analyze_all_file(self):
        self.assertEqual([], ProfilerPathManager.get_analyze_all_file(self.tmp_dir, "analyse"))
        cann_path = os.path.join(self.tmp_dir, "PROF_1_2_3a")
        analyze_path = os.path.join(cann_path, "analyze")
        os.makedirs(analyze_path)
        with os.fdopen(os.open(os.path.join(analyze_path, "test_file1.log"),
                               os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w') as fp:
            fp.write("something")
        with os.fdopen(os.open(os.path.join(analyze_path, "test_file2.log"),
                               os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w') as fp:
            fp.write("something")
        self.assertEqual(2, len(ProfilerPathManager.get_analyze_all_file(cann_path, "analyze")))

    def test_get_real_path(self):
        try:
            os.symlink("./test_link", self.tmp_dir)
            link_path = "./test_link"
        except Exception:
            link_path = ""
        if link_path:
            with self.assertRaises(RuntimeError):
                ProfilerPathManager.get_realpath(link_path)
        self.assertEqual(os.path.realpath(self.tmp_dir), ProfilerPathManager.get_realpath(self.tmp_dir))


if __name__ == "__main__":
    run_tests()
