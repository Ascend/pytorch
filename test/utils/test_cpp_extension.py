from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.utils.cpp_extension import NpuExtension


class TestCppExtension(TestCase):

    def test_npu_extension_default_libraries(self):
        extension = NpuExtension('test_extension', ['test.cpp'])
        expected_libraries = ['c10', 'torch', 'torch_npu', 'torch_python', 'torch_npu']
        for lib in expected_libraries:
            self.assertIn(lib, extension.libraries)


if __name__ == '__main__':
    run_tests()