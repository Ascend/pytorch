import random
import struct
import string
from torch_npu.profiler.analysis.prof_bean._param_tensor_bean import ParamTensorBean
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.testing.testcase import TestCase, run_tests


class TestParamTensorBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sample_num = 4
        cls.test_cases = [cls.generate_sample() for _ in range(cls.sample_num)]

    @staticmethod
    def generate_param_string():
        random_strings = []
        for _ in range(10):
            param = ''.join(random.choice(string.ascii_letters) for _ in range(10))
            random_strings.append(param)
        params = '}'.join(random_strings)
        return params

    @classmethod
    def generate_sample(cls):
        key = random.randint(0, 2**64 - 1)
        if random.choice([True, False]):
            module_params = cls.generate_param_string()
            sample = {
                Constant.CONSTANT_BYTES: struct.pack("<Q", key),
                "key": key,
                1: module_params,
            }
        else:
            optimizer_params = cls.generate_param_string()
            sample = {
                Constant.CONSTANT_BYTES: struct.pack("<Q", key),
                "key": key,
                2: optimizer_params,
            }
        return sample

    def test_property(self):
        for test_case in self.test_cases:
            param_tensor_bean = ParamTensorBean(test_case)
            self.assertEqual(test_case.get("key"), param_tensor_bean.key)
            module_params = test_case.get(1)
            optimizer_params = test_case.get(2)
            if module_params is not None:
                self.assertEqual(module_params, '}'.join(param_tensor_bean.params.module_params))
            if optimizer_params is not None:
                self.assertEqual(optimizer_params, '}'.join(param_tensor_bean.params.optimizer_params))


if __name__ == "__main__":
    run_tests()
