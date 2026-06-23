import gc
import os
import shutil
import subprocess
import unittest

import torch
import torch.utils.cpp_extension

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTORCH_INSTALL_PATH = os.path.dirname(os.path.realpath(torch.__file__))
PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.realpath(torch_npu.__file__))


def create_build_path(build_directory):
    if os.path.exists(build_directory):
        shutil.rmtree(build_directory, ignore_errors=True)
    os.makedirs(build_directory, exist_ok=True)


def build_stub(base_dir):
    build_stub_cmd = ["sh", os.path.join(base_dir, "third_party/acl/libs/build_stub.sh")]
    if subprocess.call(build_stub_cmd) != 0:
        raise RuntimeError(f"Failed to build stub: {build_stub_cmd}")


@unittest.skipIf(not torch_npu.npu.is_available(), "npu not available, skipping tests")
class TestAllocatorTraceTracker(TestCase):
    module = None
    build_directory = os.path.join(REPO_ROOT, "test", "build", "allocator_trace_tracker")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        build_stub(REPO_ROOT)
        create_build_path(cls.build_directory)

        cann_lib_path = os.path.join(REPO_ROOT, "third_party", "acl", "libs")
        torch_npu_lib_path = os.path.join(PYTORCH_NPU_INSTALL_PATH, "lib")
        extra_include_paths = [
            os.path.join(PYTORCH_NPU_INSTALL_PATH, "include"),
            os.path.join(PYTORCH_NPU_INSTALL_PATH, "include", "third_party", "acl", "inc"),
        ]
        extra_ldflags = [
            f"-L{cann_lib_path}",
            "-lascendcl",
            f"-L{torch_npu_lib_path}",
            "-ltorch_npu",
            f"-Wl,-rpath,{torch_npu_lib_path}",
            "-lc10",
            f"-L{PYTORCH_INSTALL_PATH}",
        ]

        cls.module = torch.utils.cpp_extension.load(
            name="allocator_trace_tracker_extension",
            sources=[
                os.path.join(REPO_ROOT, "test", "cpp_extensions", "allocator_trace_tracker_extension.cpp"),
            ],
            extra_include_paths=extra_include_paths,
            extra_cflags=["-g"],
            extra_ldflags=extra_ldflags,
            build_directory=cls.build_directory,
            verbose=False,
        )

        torch.empty(1, device="npu")
        cls.module.attach_trace_tracker()
        gc.collect()
        torch_npu.npu.empty_cache()

    def tearDown(self):
        self.module.reset_trace_tracker_state()
        torch_npu.npu.memory._record_memory_history(None)
        gc.collect()
        torch_npu.npu.empty_cache()
        super().tearDown()

    @staticmethod
    def _allocate_large_buffer():
        return torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="npu")

    def test_trace_tracker_callbacks_without_history(self):
        torch_npu.npu.memory._record_memory_history(None)
        self.assertFalse(torch_npu._C._npu_isHistoryEnabled())

        torch_npu.npu.empty_cache()
        self.module.reset_trace_tracker_state()

        buffer = self._allocate_large_buffer()
        state_after_alloc = self.module.get_trace_tracker_state()

        self.assertGreaterEqual(state_after_alloc["segment_alloc_count"], 1)

        del buffer
        gc.collect()
        torch_npu.npu.empty_cache()

        state_after_free = self.module.get_trace_tracker_state()
        self.assertGreaterEqual(state_after_free["segment_free_count"], 1)

    def test_trace_tracker_callbacks_with_history_enabled(self):
        torch_npu.npu.memory._record_memory_history(
            "all",
            context="alloc",
            stacks="python",
            max_entries=128,
        )
        self.assertTrue(torch_npu._C._npu_isHistoryEnabled())

        self.module.reset_trace_tracker_state()

        buffer = self._allocate_large_buffer()
        del buffer
        gc.collect()
        torch_npu.npu.empty_cache()

        state = self.module.get_trace_tracker_state()
        self.assertGreaterEqual(state["segment_alloc_count"], 1)
        self.assertGreaterEqual(state["segment_free_count"], 1)


if __name__ == "__main__":
    run_tests()
