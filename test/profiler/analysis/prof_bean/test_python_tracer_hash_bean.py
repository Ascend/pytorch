import random
import struct
import string
from torch_npu.profiler.analysis.prof_bean._python_tracer_hash_bean import PythonTracerHashBean
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.testing.testcase import TestCase, run_tests


class TestPythonTracerHashBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sample_num = 3
        cls.test_cases = [cls.generate_sample() for _ in range(cls.sample_num)]

    @classmethod
    def generate_sample(cls):
        key = random.randint(0, 2**64 - 1)
        value = ''.join(random.choice(string.ascii_letters) for _ in range(10))
        value_idx = 1
        sample = {
            Constant.CONSTANT_BYTES: struct.pack("<Q", key),
            "key": key, value_idx: value, "value": value
        }
        return sample

    def test_property(self):
        for test_case in self.test_cases:
            python_tracer_hash_bean = PythonTracerHashBean(test_case)
            self.assertEqual(test_case.get("key"), python_tracer_hash_bean.key)
            self.assertEqual(test_case.get("value"), python_tracer_hash_bean.value)


if __name__ == "__main__":
    run_tests()
