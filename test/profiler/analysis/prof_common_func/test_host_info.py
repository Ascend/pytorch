from torch_npu.profiler.analysis.prof_common_func._host_info import get_host_info

from torch_npu.testing.testcase import TestCase, run_tests


class TestHostInfo(TestCase):

    def test_get_host_info(self):
        _host_info = get_host_info()
        self.assertNotEqual('0', _host_info.get('host_uid'))


if __name__ == "__main__":
    run_tests()
