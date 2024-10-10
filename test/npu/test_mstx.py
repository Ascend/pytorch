import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestMstx(TestCase):
    mark_msg = ''
    range_msg = ''
    range_id = 0

    def setUp(self):
        def stub_mark(message: str = ''):
            self.mark_msg = message

        def stub_range_start_on_host(message: str) -> int:
            self.range_msg = message
            self.range_id += 1
            return self.range_id

        def stub_range_start(message: str, stream=None):
            self.range_msg = message
            self.range_id += 1
            return self.range_id

        def stub_range_end(range_id: int):
            self.range_id = range_id

        torch_npu._C._mark = stub_mark
        torch_npu._C._mstx._range_start = stub_range_start
        torch_npu._C._mstx._range_start_on_host = stub_range_start_on_host
        torch_npu._C._mstx._range_end = stub_range_end

    def test_mark(self):
        torch_npu.npu.mstx.mark("test1")
        self.assertEqual("test1", self.mark_msg)
        torch_npu.npu.mstx().mark("test2") # Verify compatibility
        self.assertEqual("test2", self.mark_msg)

    def test_range_start(self):
        self.range_id = 0
        ret_id = torch_npu.npu.mstx.range_start("")
        self.assertEqual(0, ret_id)
        ret_id = torch_npu.npu.mstx.range_start("test1")
        self.assertEqual(1, ret_id)
        self.assertEqual("test1", self.range_msg)
        ret_id = torch_npu.npu.mstx.range_start("test2", None)
        self.assertEqual(2, ret_id)
        self.assertEqual("test2", self.range_msg)
        
        torch.npu.set_device(0)
        current_stream = torch.npu.current_stream()
        ret_id = torch_npu.npu.mstx.range_start("test3", current_stream)
        self.assertEqual(3, ret_id)
        self.assertEqual("test3", self.range_msg)
        ret_id = torch_npu.npu.mstx.range_start("test4", 'invalid_stream')
        self.assertEqual(0, ret_id)

    def test_range_end(self):
        self.range_id = 0
        torch_npu.npu.mstx.range_end('invalid_range_id')
        self.assertEqual(0, self.range_id)
        torch_npu.npu.mstx.range_end(1)
        self.assertEqual(1, self.range_id)


if __name__ == '__main__':
    run_tests()