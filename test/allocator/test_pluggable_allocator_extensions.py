import os
import sys
import shutil
import subprocess
import ctypes
import torch
import torch.utils.cpp_extension

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


PYTORCH_INSTALL_PATH = os.path.dirname(os.path.realpath(torch.__file__))
PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.realpath(torch_npu.__file__))


def create_build_path(build_directory):
    if os.path.exists(build_directory):
        shutil.rmtree(build_directory, ignore_errors=True)
    os.makedirs(build_directory, exist_ok=True)


def build_stub(base_dir):
    build_stub_cmd = ["sh", os.path.join(base_dir, 'third_party/acl/libs/build_stub.sh')]
    if subprocess.call(build_stub_cmd) != 0:
        raise RuntimeError('Failed to build stub: {}'.format(build_stub_cmd))


class TestPluggableAllocator(TestCase):
    module = None
    new_alloc = None
    build_directory = "allocator/build"

    @classmethod
    def setUpClass(cls):
        # Build Extension
        BASE_DIR = os.path.abspath("./../")
        build_stub(BASE_DIR)
        create_build_path(cls.build_directory)
        CANN_LIB_PATH = os.path.join(BASE_DIR, 'third_party/acl/libs')
        extra_ldflags = []
        extra_ldflags.append("-lascendcl")
        extra_ldflags.append(f"-L{CANN_LIB_PATH}")
        extra_ldflags.append("-lc10")
        extra_ldflags.append(f"-L{PYTORCH_INSTALL_PATH}")
        extra_include_paths = ["cpp_extensions"]
        extra_include_paths.append(os.path.join(PYTORCH_NPU_INSTALL_PATH, 'include'))

        cls.module = torch.utils.cpp_extension.load(
            name="pluggable_allocator_extensions",
            sources=[
                "cpp_extensions/pluggable_allocator_extensions.cpp"
            ],
            extra_include_paths=extra_include_paths,
            extra_cflags=["-g"],
            extra_ldflags=extra_ldflags,
            build_directory=cls.build_directory,
            verbose=True,
        )
    
    def test_pluggable_allocator(self):
        os_path = os.path.join(TestPluggableAllocator.build_directory, 'pluggable_allocator_extensions.so')
        # Load the allocator
        TestPluggableAllocator.new_alloc = torch_npu.npu.memory.NPUPluggableAllocator(os_path, 'my_malloc', 'my_free')
        # Swap the current allocator
        torch_npu.npu.memory.change_current_allocator(TestPluggableAllocator.new_alloc)
        # This will allocate memory in the device using the new allocator
        self.assertFalse(self.module.check_custom_allocator_used())
        npu_tensor = torch.zeros(10, device='npu')
        cpu_tensor = torch.zeros(10)
        self.assertRtolEqual(npu_tensor.cpu().numpy(), cpu_tensor.numpy())
        self.assertTrue(self.module.check_custom_allocator_used())

    def test_set_get_device_stats_fn(self):
        os_path = os.path.join(TestPluggableAllocator.build_directory, 'pluggable_allocator_extensions.so')
        myallocator = ctypes.CDLL(os_path)
        get_device_stats_fn = ctypes.cast(getattr(myallocator, "my_get_device_stats"), ctypes.c_void_p).value

        TestPluggableAllocator.new_alloc.allocator().set_get_device_stats_fn(get_device_stats_fn)
        self.assertEqual(torch.npu.memory_stats_as_nested_dict()["num_alloc_retries"], 0)

    def test_set_reset_peak_status_fn(self):
        os_path = os.path.join(TestPluggableAllocator.build_directory, 'pluggable_allocator_extensions.so')
        myallocator = ctypes.CDLL(os_path)
        reset_peak_status_fn = ctypes.cast(getattr(myallocator, "my_reset_peak_status"), ctypes.c_void_p).value

        TestPluggableAllocator.new_alloc.allocator().set_reset_peak_status_fn(reset_peak_status_fn)
        torch.npu.reset_peak_memory_stats()
        self.assertEqual(torch.npu.max_memory_allocated(), 0)

    def test_pluggable_allocator_after_init(self):
        os_path = os.path.join(TestPluggableAllocator.build_directory, 'pluggable_allocator_extensions.so')
        # Do an initial memory allocator
        ori_tensor = torch.zeros(10, device='npu')
        # Load the allocator
        new_alloc = torch_npu.npu.memory.NPUPluggableAllocator(os_path, 'my_malloc', 'my_free')
        msg = "Can't swap an already initialized allocator"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch_npu.npu.memory.change_current_allocator(new_alloc)


if __name__ == "__main__":
    run_tests()
