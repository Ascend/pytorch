import os
import shutil
import stat
import json

from torch_npu.profiler.analysis.prof_bean._ge_memory_record_bean import GeMemoryRecordBean
from torch_npu.profiler.analysis.prof_common_func._file_manager import FileManager

from torch_npu.testing.testcase import TestCase, run_tests


class TestFileManager(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp_dir = "./tmp_dir"
        os.makedirs(cls.tmp_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_dir)

    def test_file_all(self):
        test_file_path = os.path.join(self.tmp_dir, "test_file.log")
        with os.fdopen(os.open(test_file_path,
                               os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w') as fp:
            fp.write("something")
        self.assertEqual("something", FileManager.file_read_all(test_file_path))

    def test_read_csv_file(self):
        dir_path = self.tmp_dir
        test_file1 = os.path.join(self.tmp_dir, "test_file1.csv")
        with os.fdopen(os.open(test_file1,
                               os.O_WRONLY | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR), 'w') as fp:
            fp.write("Component,Timestamp(us),Total Allocated(KB),Total Reserved(KB),Device\n")
            fp.write("APP,18.927,1024,2048,NPU:0\n")
        self.assertEqual([], FileManager.read_csv_file(dir_path, GeMemoryRecordBean))
        test_dict = {
            "Component": "APP", "Timestamp(us)": "18.927", "Total Allocated(KB)":1024,
            "Total Reserved(KB)": 2048, "Device": "NPU:0"
        }
        expect = GeMemoryRecordBean(test_dict)
        read_result = FileManager.read_csv_file(test_file1, GeMemoryRecordBean)
        self.assertEqual(1, len(read_result))
        self.assertEqual(expect.component, read_result[0].component)
        self.assertEqual(expect.device_tag, read_result[0].device_tag)
        self.assertEqual(expect.time_ns, read_result[0].time_ns)
        self.assertEqual(expect.total_allocated, read_result[0].total_allocated)
        self.assertEqual(expect.total_reserved, read_result[0].total_reserved)

    def test_create_csv_file(self):
        test_file = "test_file.csv"
        headers = ["H1", "H2", "H3"]
        data = [["header10", "header11", "header12"]]
        FileManager.create_csv_file(self.tmp_dir, data, test_file, headers)
        test_file_path = os.path.join(self.tmp_dir, test_file)
        with open(test_file_path, 'r') as fp:
            read_header = fp.readline()
            self.assertEqual("H1,H2,H3\n", read_header)
            line1 = fp.readline()
            self.assertEqual("header10,header11,header12\n", line1)

    def test_create_json_file_by_path(self):
        test_file = "test_file.json"
        data = {"Name":"ZhangShan", "Age":666}
        output_path = os.path.join(self.tmp_dir, test_file)
        FileManager.create_json_file_by_path(output_path, data)
        with open(output_path, 'r') as fp:
            read_data = json.load(fp)
        self.assertEqual(data, read_data)

    def test_append_trace(self):
        test_file = "test_file.json"
        data1 = {"Name":"ZhangShan", "Age":666}
        data2 = {"Height":180, "Addr":"China"}
        output_path = os.path.join(self.tmp_dir, test_file)
        FileManager.create_prepare_trace_json_by_path(output_path, data1)
        FileManager.append_trace_json_by_path(output_path, data2, output_path)
        with open(output_path, 'r') as fp:
            read_data = json.load(fp)
        expect = {**data1, **data2}
        self.assertEqual(read_data, expect)


if __name__ == "__main__":
    run_tests()
