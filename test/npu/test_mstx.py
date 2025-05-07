import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestMstx(TestCase):
    mark_msg = ''
    mark_domain = ''
    range_msg = ''
    range_id = 0
    range_domain = ''

    def setUp(self):
        def stub_mark(message: str, stream=None, domain: str = 'default'):
            self.mark_msg = message
            self.mark_domain = domain

        def stub_mark_on_host(message: str, domain: str = 'default'):
            self.mark_msg = message
            self.mark_domain = domain

        def stub_range_start_on_host(message: str, domain: str = 'default') -> int:
            self.range_msg = message
            self.range_id += 1
            self.range_domain = domain
            return self.range_id

        def stub_range_start(message: str, stream=None, domain: str = 'default'):
            self.range_msg = message
            self.range_id += 1
            self.range_domain = domain
            return self.range_id

        def stub_range_end(range_id: int, domain: str = 'default'):
            self.range_id = range_id
            self.range_domain = domain

        torch_npu._C._mstx._mark = stub_mark
        torch_npu._C._mstx._mark_on_host = stub_mark_on_host
        torch_npu._C._mstx._range_start = stub_range_start
        torch_npu._C._mstx._range_start_on_host = stub_range_start_on_host
        torch_npu._C._mstx._range_end = stub_range_end

    def test_mark(self):
        # invalid inputs
        torch_npu.npu.mstx.mark("")
        self.assertEqual("", self.mark_msg)
        self.assertEqual("", self.mark_domain)
        torch_npu.npu.mstx.mark(message=0)
        self.assertEqual("", self.mark_msg)
        self.assertEqual("", self.mark_domain)
        torch_npu.npu.mstx.mark("test", stream=None, domain=1)
        self.assertEqual("", self.mark_msg)
        self.assertEqual("", self.mark_domain)
        torch_npu.npu.mstx.mark("test", stream=1, domain="test")
        self.assertEqual("", self.mark_msg)
        self.assertEqual("", self.mark_domain) 

        # valid inputs
        torch_npu.npu.mstx.mark("test1")
        self.assertEqual("test1", self.mark_msg)
        self.assertEqual("default", self.mark_domain)
        torch_npu.npu.mstx.mark("test2", stream=None, domain="test_domain1")
        self.assertEqual("test2", self.mark_msg)
        self.assertEqual("test_domain1", self.mark_domain)

        torch.npu.set_device(0)
        current_stream = torch.npu.current_stream()
        torch_npu.npu.mstx.mark("test3", stream=current_stream, domain="test_domain2")
        self.assertEqual("test3", self.mark_msg)
        self.assertEqual("test_domain2", self.mark_domain)


    def test_range_start(self):
        # invalid inputs
        ret_id = torch_npu.npu.mstx.range_start("")
        self.assertEqual(0, ret_id)
        ret_id = torch_npu.npu.mstx.range_start(message=0)
        self.assertEqual(0, ret_id)
        ret_id = torch_npu.npu.mstx.range_start(message="test", stream=None, domain=1)
        self.assertEqual(0, ret_id)
        ret_id = torch_npu.npu.mstx.range_start(message="test", stream=1, domain="test")
        self.assertEqual(0, ret_id)

        # valid inputs
        ret_id = torch_npu.npu.mstx.range_start("test1")
        self.assertEqual(1, ret_id)
        self.assertEqual("test1", self.range_msg)
        self.assertEqual("default", self.range_domain)
        ret_id = torch_npu.npu.mstx.range_start("test2", None)
        self.assertEqual(2, ret_id)
        self.assertEqual("test2", self.range_msg)
        self.assertEqual("default", self.range_domain)
        ret_id = torch_npu.npu.mstx.range_start("test3", None, domain="test_domain1")
        self.assertEqual(3, ret_id)
        self.assertEqual("test3", self.range_msg)
        self.assertEqual("test_domain1", self.range_domain)

        torch.npu.set_device(0)
        current_stream = torch.npu.current_stream()
        ret_id = torch_npu.npu.mstx.range_start("test4", current_stream)
        self.assertEqual(4, ret_id)
        self.assertEqual("test4", self.range_msg)
        self.assertEqual("default", self.range_domain)
        ret_id = torch_npu.npu.mstx.range_start("test5", current_stream, domain="test_domain2")
        self.assertEqual(5, ret_id)
        self.assertEqual("test5", self.range_msg)
        self.assertEqual("test_domain2", self.range_domain)

    def test_range_end(self):
        self.range_id = 0
        torch_npu.npu.mstx.range_end('invalid_range_id')
        self.assertEqual(0, self.range_id)
        torch_npu.npu.mstx.range_end(1)
        self.assertEqual(1, self.range_id)
        self.assertEqual("default", self.range_domain)
        torch_npu.npu.mstx.range_end(2, domain="test_domain1")
        self.assertEqual(2, self.range_id)
        self.assertEqual("test_domain1", self.range_domain)


if __name__ == '__main__':
    run_tests()