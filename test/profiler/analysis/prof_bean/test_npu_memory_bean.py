import random

from torch_npu.profiler.analysis.prof_bean._npu_mem_bean import NpuMemoryBean
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuMemoryBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sample_num = 20
        cls.samples = cls.generate_samples()

    @classmethod
    def generate_samples(cls):
        samples = []
        for _ in range(cls.sample_num):
            event = random.choice([Constant.APP, Constant.PTA, Constant.GE, Constant.PTA_GE])
            ts = random.randint(0, 2**31)
            alloc = random.choice([0, 1024, 2048, 4096])
            mem = random.choice([0, 1024, 2048, 4096])
            device_id = random.randint(0, 2**31)
            sample = {
                "event": str(event),
                "timestamp(us)": str(ts),
                "allocated(KB)": str(alloc),
                "memory(KB)": str(mem),
                "Device_id": str(device_id)
            }
            samples.append(sample)
        return samples

    def test_property(self):
        for sample in self.samples:
            npu_mem_bean = NpuMemoryBean(sample)
            if sample.get("event") != Constant.APP:
                row = []
            else:
                row = [sample.get("event"), sample.get("timestamp(us)"),
                       sample.get("allocated(KB)"), float(sample.get("memory(KB)")) / Constant.KB_TO_MB,
                       "", "", sample.get("Device_id")]
            self.assertEqual(row, npu_mem_bean.row)


if __name__ == "__main__":
    run_tests()
