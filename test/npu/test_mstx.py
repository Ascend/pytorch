import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestMstx(TestCase):
    mark_msg = ''
    mark_domain = ''
    range_msg = ''
    range_id = 0
    range_domain = ''
    push_depth = {}

    def setUp(self):
        def stub_mark(message: str, stream_id: int = 0, device_index: int = 0,
                      device_type: int = 0, domain: str = 'default'):
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

        def stub_range_start(message: str, stream_id: int = 0, device_index: int = 0,
                             device_type: int = 0, domain: str = 'default'):
            self.range_msg = message
            self.range_id += 1
            self.range_domain = domain
            return self.range_id

        def stub_range_end(range_id: int, domain: str = 'default'):
            self.range_id = range_id
            self.range_domain = domain

        def stub_range_push(message: str, stream_id: int = 0, device_index: int = 0,
                             device_type: int = 0, domain: str = 'default') -> int:
            # For simplicity of testing, we use a dict to track the push depth for each domain.
            if domain in self.push_depth.keys():
                self.push_depth[domain].append(message)
            else:
                self.push_depth[domain] = [message]
            return len(self.push_depth[domain]) - 1

        def stub_range_push_on_host(message: str, domain: str = 'default') -> int:
            if domain in self.push_depth.keys():
                self.push_depth[domain].append(message)
            else:
                self.push_depth[domain] = [message]
            return len(self.push_depth[domain]) - 1

        def stub_range_pop(domain: str = 'default') -> int:
            if domain in self.push_depth.keys():
                depth = len(self.push_depth[domain]) - 1
                if len(self.push_depth[domain]) > 0:
                    self.push_depth[domain].pop()
                return depth
            else:
                return -1

        torch_npu._C._mstx._mark = stub_mark
        torch_npu._C._mstx._mark_on_host = stub_mark_on_host
        torch_npu._C._mstx._range_start = stub_range_start
        torch_npu._C._mstx._range_start_on_host = stub_range_start_on_host
        torch_npu._C._mstx._range_end = stub_range_end
        torch_npu._C._mstx._range_push = stub_range_push
        torch_npu._C._mstx._range_push_on_host = stub_range_push_on_host
        torch_npu._C._mstx._range_pop = stub_range_pop

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

    def test_range_push_will_return_err_value_when_called_with_invalid_inputs(self):
        # invalid inputs
        ret_id = torch_npu.npu.mstx.range_push("")
        self.assertEqual(-1, ret_id)
        ret_id = torch_npu.npu.mstx.range_push(message=0)
        self.assertEqual(-1, ret_id)
        ret_id = torch_npu.npu.mstx.range_push(message="test", stream=None, domain=1)
        self.assertEqual(-1, ret_id)
        ret_id = torch_npu.npu.mstx.range_push(message="test", stream=1, domain="test")
        self.assertEqual(-1, ret_id)

    def test_range_push_will_return_increased_depth_when_called_with_valid_inputs(self):
        # valid inputs
        ret_id = torch_npu.npu.mstx.range_push("test1")
        self.assertEqual(0, ret_id)
        self.assertEqual({"default": ["test1"]}, self.push_depth)
        ret_id = torch_npu.npu.mstx.range_push("test2", None, domain="test_domain1")
        self.assertEqual(0, ret_id)
        self.assertEqual({"default": ["test1"], "test_domain1": ["test2"]}, self.push_depth)
        ret_id = torch_npu.npu.mstx.range_push("test3", None, domain="test_domain1")
        self.assertEqual(1, ret_id)
        self.assertEqual({"default": ["test1"], "test_domain1": ["test2", "test3"]}, self.push_depth)
        torch.npu.set_device(0)
        current_stream = torch.npu.current_stream()
        ret_id = torch_npu.npu.mstx.range_push("test4", current_stream, domain="test_domain2")
        self.assertEqual(0, ret_id)
        self.assertEqual({"default": ["test1"], "test_domain1": ["test2", "test3"], "test_domain2": ["test4"]}, self.push_depth)
        ret_id = torch_npu.npu.mstx.range_push("test5", current_stream, domain="test_domain2")
        self.assertEqual(1, ret_id)
        self.assertEqual({"default": ["test1"], "test_domain1": ["test2", "test3"], "test_domain2": ["test4", "test5"]}, self.push_depth)

    def test_range_pop_will_return_err_value_when_called_with_invalid_inputs(self):
        # invalid inputs
        ret_id = torch_npu.npu.mstx.range_pop(domain=1)
        self.assertEqual(-1, ret_id)
        ret_id = torch_npu.npu.mstx.range_pop(domain="")
        self.assertEqual(-1, ret_id)

    def test_range_pop_will_return_decreased_depth_when_called_after_range_push_is_called(self):
        # valid inputs
        self.push_depth = {}
        torch_npu.npu.mstx.range_push("test1", None, domain="test_domain1")
        torch_npu.npu.mstx.range_push("test2", None, domain="test_domain1")
        ret_id = torch_npu.npu.mstx.range_pop(domain="test_domain1")
        self.assertEqual(1, ret_id)
        self.assertEqual({"test_domain1": ["test1"]}, self.push_depth)
        ret_id = torch_npu.npu.mstx.range_pop(domain="test_domain1")
        self.assertEqual(0, ret_id)
        self.assertEqual({"test_domain1": []}, self.push_depth)
        ret_id = torch_npu.npu.mstx.range_pop(domain="test_domain1")
        self.assertEqual(-1, ret_id)
        self.assertEqual({"test_domain1": []}, self.push_depth)

if __name__ == '__main__':
    run_tests()