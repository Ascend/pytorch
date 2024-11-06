import struct

from torch_npu.profiler.analysis.prof_bean._op_mark_bean import OpMarkBean
from torch_npu.profiler.analysis.prof_common_func._tlv_decoder import TLVDecoder
from torch_npu.testing.testcase import TestCase, run_tests


class TestTLVDecoder(TestCase):

    def setUp(self):
        self.data_map = {
            "time_ns": 111, "category": 2, "corr_id": 333, "tid": 444,
            "pid": 555, "name": "test", "struct_size": 40
        }
        self.encoded_data = struct.pack(
            "<HIq4QHI4c", 1, 50, 111, 2, 333, 444, 555, 1, 4, b"t", b"e", b"s", b"t"
        )

    def test_decode(self):
        op_mark_list = TLVDecoder.decode(
            self.encoded_data, OpMarkBean, self.data_map.get("struct_size"))
        self.assertEqual(1, len(op_mark_list))
        op_mark_bean = op_mark_list[0]
        self.assertEqual(f"Dequeue@{self.data_map.get('name')}", op_mark_bean.name)
        self.assertEqual(self.data_map.get("corr_id"), op_mark_bean.corr_id)
        self.assertEqual(self.data_map.get("tid"), op_mark_bean.tid)
        self.assertEqual(self.data_map.get("pid"), op_mark_bean.pid)


if __name__ == "__main__":
    run_tests()
