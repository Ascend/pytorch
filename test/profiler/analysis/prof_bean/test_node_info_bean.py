import random
from collections import namedtuple

from torch_npu.profiler.analysis.prof_bean._node_info_bean import NodeInfoBean
from torch_npu.profiler.analysis.prof_common_func._constant import convert_us2ns
from torch_npu.testing.testcase import TestCase, run_tests


class TestNodeInfoBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.kernel_num = 3
        cls.sample_num = 3
        cls.samples = cls.generate_samples()

    @classmethod
    def generate_samples(cls):
        samples = []
        for _ in range(cls.sample_num):
            kernel_list = []
            Kernel = namedtuple("Kernel", ["ts", "dur", "is_ai_core"])
            for _ in range(cls.kernel_num):
                ts = random.randint(0, 2**31)
                dur = random.random() * 100
                is_ai_core = True if random.choice([0, 1]) else False
                kernel = Kernel(ts, dur, is_ai_core)
                kernel_list.append(kernel)
            samples.append(kernel_list)
        return samples

    def test_property(self):
        for sample in self.samples:
            node_indo_bean = NodeInfoBean(sample)
            device_dur = sum([float(kernel.dur) for kernel in sample])
            device_dur_with_ai_core = sum([float(kernel.dur) for kernel in sample if kernel.is_ai_core])
            min_start = min([kernel.ts for kernel in sample])
            max_end = max([kernel.ts + convert_us2ns(kernel.dur) for kernel in sample])
            self.assertEqual(sample, node_indo_bean.kernel_list)
            self.assertEqual(device_dur, node_indo_bean.device_dur)
            self.assertEqual(device_dur_with_ai_core, node_indo_bean.device_dur_with_ai_core)
            self.assertEqual(min_start, node_indo_bean.min_start)
            self.assertEqual(max_end, node_indo_bean.max_end)


if __name__ == "__main__":
    run_tests()
