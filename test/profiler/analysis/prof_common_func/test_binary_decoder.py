import torch_npu.npu._sanitizer as sanitizer
import torch_npu.profiler.analysis.prof_common_func._binary_decoder as binary_decoder
from torch_npu.testing.testcase import TestCase, run_tests


class TestBinaryDecoder(TestCase):
    def test_decode_multiple_structures(self):
        class MockBean:
            def __init__(self, data):
                self.data = data

            def __eq__(self, other):
                return isinstance(other, MockBean) and self.data == other.data

        test_bytes = b'\x01\x02\x03\x04\x05\x06'
        result = binary_decoder.BinaryDecoder.decode(test_bytes, MockBean, 2)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].data, b'\x01\x02')
        self.assertEqual(result[1].data, b'\x03\x04')
        self.assertEqual(result[2].data, b'\x05\x06')


if __name__ == "__main__":
    run_tests()