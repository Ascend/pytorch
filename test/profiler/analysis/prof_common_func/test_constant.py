from torch_npu.profiler.analysis.prof_common_func._constant import (
    convert_ns2us_float, convert_us2ns, convert_ns2us_str, contact_2num
)

from torch_npu.testing.testcase import TestCase, run_tests


class TestConstant(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_convert_ns2us_float(self):
        self.assertEqual(True, convert_ns2us_float(float("inf")) == float("inf"))
        with self.assertRaises((RuntimeError, TypeError)):
            convert_ns2us_float("17")
        with self.assertRaises((RuntimeError, TypeError)):
            convert_ns2us_float(18789.89)
        self.assertEqual(str(convert_ns2us_float(1459635878536856678)), str(1459635878536856678 / 1000))
    
    def test_convert_ns2us_str(self):
        self.assertEqual(convert_ns2us_str(float("inf")), "inf")
        with self.assertRaises((RuntimeError, TypeError)):
            convert_ns2us_str("17")
        with self.assertRaises((RuntimeError, TypeError)):
            convert_ns2us_float(18789.89)
        self.assertEqual(convert_ns2us_str(1459635878536856678), "1459635878536856.678")
        self.assertEqual(convert_ns2us_str(1459635878536856678, "\t"), "1459635878536856.678\t")

    def test_convert_us2ns(self):
        self.assertEqual(convert_us2ns("1459635878536856.678\t"), 1459635878536856678)
        self.assertEqual(convert_us2ns("1459635878536856.678"), 1459635878536856678)
        self.assertEqual(convert_us2ns(1459635878536856.678), 1459635878536856800)

    def test_contact_2num(self):
        high_num = 1
        low_num = 1
        result = 1 << 32 | 1
        self.assertEqual(contact_2num(high_num, low_num), result)


if __name__ == "__main__":
    run_tests()
