# Owner(s): ["module: unknown"]
import os
import platform
import shutil
import subprocess
import unittest

import torch
import torch.utils.cpp_extension

import torch_npu
from torch_npu.testing.testcase import run_tests, TestCase


PYTORCH_INSTALL_PATH = os.path.dirname(os.path.realpath(torch.__file__))
PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.realpath(torch_npu.__file__))
IS_ARM64 = platform.machine() in ('arm64', 'aarch64')


def setup_sanitizer():
    os.environ["TORCH_NPU_SANITIZER"] = "1"
    import torch_npu.npu._sanitizer as sanitizer

    if not sanitizer.npu_sanitizer.enabled:
        sanitizer.npu_sanitizer.enable()


def get_event_handler():
    import torch_npu.npu._sanitizer as sanitizer

    return sanitizer.npu_sanitizer.event_handler


def create_build_path(build_directory):
    if os.path.exists(build_directory):
        shutil.rmtree(build_directory, ignore_errors=True)
    os.makedirs(build_directory, exist_ok=True)


def build_stub(base_dir):
    build_stub_cmd = [
        "sh",
        os.path.join(base_dir, "third_party/acl/libs/build_stub.sh"),
    ]
    if subprocess.call(build_stub_cmd) != 0:
        raise RuntimeError(f"Failed to build stub: {build_stub_cmd}")


def reset_sanitizer():
    import torch_npu.npu._sanitizer as sanitizer

    if sanitizer.npu_sanitizer.dispatch is not None:
        try:
            sanitizer.npu_sanitizer.dispatch.__exit__(None, None, None)
        except Exception:
            pass
        sanitizer.npu_sanitizer.dispatch = None

    sanitizer.npu_sanitizer.event_handler = None
    sanitizer.npu_sanitizer.enabled = False


@unittest.skipUnless(IS_ARM64, "Only working on ARM")
class TestSanitizerPluggableAllocator(TestCase):
    module = None
    build_directory = os.path.join("allocator", "build_sanitizer_pluggable")

    @classmethod
    def setUpClass(cls):
        BASE_DIR = os.path.abspath("./../")
        build_stub(BASE_DIR)
        create_build_path(cls.build_directory)
        CANN_LIB_PATH = os.path.join(BASE_DIR, "third_party/acl/libs")
        extra_ldflags = []
        extra_ldflags.append("-lascendcl")
        extra_ldflags.append(f"-L{CANN_LIB_PATH}")
        extra_ldflags.append("-lc10")
        extra_ldflags.append(f"-L{PYTORCH_INSTALL_PATH}")
        extra_include_paths = ["cpp_extensions"]
        extra_include_paths.append(os.path.join(PYTORCH_NPU_INSTALL_PATH, "include"))

        cls.module = torch.utils.cpp_extension.load(
            name="sanitizer_pluggable_allocator_extensions",
            sources=["cpp_extensions/pluggable_allocator_extensions.cpp"],
            extra_include_paths=extra_include_paths,
            extra_cflags=["-g"],
            extra_ldflags=extra_ldflags,
            build_directory=cls.build_directory,
            verbose=True,
        )

    def test_pluggable_allocator_record_stream_warning_and_suppression(self):
        """Pluggable allocator should support both missing-record_stream warning and suppression."""
        os_path = os.path.join(
            self.build_directory, "sanitizer_pluggable_allocator_extensions.so"
        )
        allocator = torch_npu.npu.memory.NPUPluggableAllocator(
            os_path, "my_malloc", "my_free"
        )
        torch_npu.npu.memory.change_current_allocator(allocator)

        # Case 1: no record_stream -> should warn, guards false negative.
        setup_sanitizer()

        x = torch.randn(100, device="npu")
        stream1 = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        stream1.wait_stream(default_stream)
        with torch_npu.npu.stream(stream1):
            _ = x + 1

        warnings = get_event_handler().flush_record_stream_warnings()
        self.assertGreater(
            len(warnings),
            0,
            "Missing record_stream should be detected with NPUPluggableAllocator.",
        )

        reset_sanitizer()

        # Case 2: record_stream -> should not warn, guards false positive.
        setup_sanitizer()

        y = torch.randn(100, device="npu")
        stream2 = torch_npu.npu.Stream()
        default_stream = torch_npu.npu.default_stream()

        y.record_stream(stream2)
        stream2.wait_stream(default_stream)
        with torch_npu.npu.stream(stream2):
            _ = y + 1

        warnings = get_event_handler().flush_record_stream_warnings()
        self.assertEqual(
            len(warnings),
            0,
            "record_stream should suppress missing-record_stream warning with NPUPluggableAllocator.",
        )

if __name__ == "__main__":
    run_tests()
