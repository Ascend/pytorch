from collections import OrderedDict

from torch_npu.profiler.analysis.prof_bean._ai_cpu_bean import AiCpuBean
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.profiler.analysis.prof_bean._hccs_bean import HccsBean
from torch_npu.profiler.analysis.prof_bean._nic_bean import NicBean
from torch_npu.profiler.analysis.prof_bean._npu_module_mem_bean import NpuModuleMemoryBean
from torch_npu.profiler.analysis.prof_bean._pcie_bean import PcieBean
from torch_npu.profiler.analysis.prof_bean._roce_bean import RoCEBean


class TestAiCPUBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        valid_data = OrderedDict()
        valid_data["Timestamps(us)"] = 1768
        valid_data["Node"] = "IndexPutV2"
        valid_data["Compute_time(us)"] = 0.2
        cls.test_cases = [
            # data, keys, values, exception
            ["string_data", None, None, AttributeError],
            [[1, 2, 3], None, None, AttributeError],
            [valid_data, ["Timestamps(us)", "Node", "Compute_time(us)"],
             [1768, "IndexPutV2", 0.2], None],
        ]

    def test_row(self):
        for test_case in self.test_cases:
            data, _, values, exception = test_case
            _ai_cpu_bean = AiCpuBean(data)
            if exception:
                with self.assertRaises(exception):
                    _ = _ai_cpu_bean.row
                continue
            self.assertEqual(values, _ai_cpu_bean.row)

    def test_headers(self):
        for test_case in self.test_cases:
            data, keys, _, exception = test_case
            _ai_cpu_bean = AiCpuBean(data)
            if exception:
                with self.assertRaises(exception):
                    _ = _ai_cpu_bean.headers
                continue
            self.assertEqual(keys, _ai_cpu_bean.headers)

    def test_hccs_bean_constructor_initialization(self):
        valid_data = OrderedDict()
        valid_data["Timestamps(us)"] = 1768
        valid_data["Node"] = "IndexPutV2"
        valid_data["Compute_time(us)"] = 0.2
        hccs_bean = HccsBean(valid_data)
        self.assertEqual([1768, "IndexPutV2", 0.2], hccs_bean.row)
        self.assertEqual(
            ["Timestamps(us)", "Node", "Compute_time(us)"],
            hccs_bean.headers
        )

    def test_nic_bean_constructor_initialization(self):
        valid_data = OrderedDict()
        valid_data["Timestamps(us)"] = 1768
        valid_data["Node"] = "IndexPutV2"
        valid_data["Compute_time(us)"] = 0.2
        nic_bean = NicBean(valid_data)
        self.assertEqual([1768, "IndexPutV2", 0.2], nic_bean.row)
        self.assertEqual(
            ["Timestamps(us)", "Node", "Compute_time(us)"],
            nic_bean.headers
        )

    def test_npu_module_memory_bean_headers_property(self):
        valid_data = OrderedDict()
        valid_data["Device_id"] = "3"
        valid_data["Component"] = "TestComponent4"
        valid_data["Timestamp(us)"] = "111111"
        valid_data["Total Reserved(KB)"] = "512000"
        valid_data["Device"] = "NPU3"

        npu_bean = NpuModuleMemoryBean(valid_data)
        result_headers = npu_bean.headers
        excepted_headers = [
            "Device_id", "Component", "Timestamp(us)", "Total Reserved(MB)", "Device"
        ]

        self.assertEqual(result_headers, excepted_headers)

    def test_npu_module_memory_bean_row_property(self):
        valid_data = OrderedDict()
        valid_data["Device_id"] = "2"
        valid_data["Component"] = "TestComponent3"
        valid_data["Timestamp(us)"] = "987654"
        valid_data["Total Reserved(KB)"] = "1024000"
        valid_data["Device"] = "NPU2"

        npu_bean = NpuModuleMemoryBean(valid_data)
        result_row = npu_bean.row
        excepted_row = [
            "2", "TestComponent3", "987654", "1000.0", "NPU2"
        ]

        self.assertEqual(result_row, excepted_row)

    def test_pcie_bean_constructor_initialization(self):
        valid_data = OrderedDict()
        valid_data["Timestamps(us)"] = 1768
        valid_data["Node"] = "IndexPutV2"
        valid_data["Compute_time(us)"] = 0.2
        pcie_bean = PcieBean(valid_data)
        self.assertEqual([1768, "IndexPutV2", 0.2], pcie_bean.row)
        self.assertEqual(
            ["Timestamps(us)", "Node", "Compute_time(us)"],
            pcie_bean.headers
        )

    def test_roce_bean_constructor_initialization(self):
        valid_data = OrderedDict()
        valid_data["Timestamps(us)"] = 1768
        valid_data["Node"] = "IndexPutV2"
        valid_data["Compute_time(us)"] = 0.2
        roce_bean = RoCEBean(valid_data)
        self.assertEqual([1768, "IndexPutV2", 0.2], roce_bean.row)
        self.assertEqual(
            ["Timestamps(us)", "Node", "Compute_time(us)"],
            roce_bean.headers
        )


if __name__ == "__main__":
    run_tests()
