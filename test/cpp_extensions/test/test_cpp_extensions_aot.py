import os
import stat
import pathlib
import subprocess
import unittest

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

try:
    import torch_test_cpp_extension.npu as npu_extension
except ImportError as e:
    raise RuntimeError(
        "test_cpp_extensions_aot.py cannot be invoked directly. Run "
        "`python run_cpp_test.py` instead.") from e


class TestCppExtensionAOT(TestCase):
    """Tests ahead-of-time cpp extensions
    """

    def test_npu_extension(self):
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        z = npu_extension.tanh_add(x, y)
        self.assertEqual(z, x.tanh() + y.tanh())

        z = npu_extension.tanh_add(x.npu(), y.npu())
        expect_out = x.npu().tanh() + y.npu().tanh()
        self.assertEqual(z.cpu(), expect_out.cpu())

        npu_z = npu_extension.npu_add(x.npu(), y.npu())
        self.assertEqual(npu_z.cpu(), (x + y))

    def test_storage_sizes(self):
        t = torch_npu.npu_format_cast(torch.ones(128, 512, dtype=torch.int8).npu(), 29)
        self.assertTrue(npu_extension.check_storage_sizes(t, (16, 8, 16, 32)))
        t = torch_npu.npu_format_cast(torch.ones(31, 127, 511, dtype=torch.int8).npu(), 29)
        self.assertTrue(npu_extension.check_storage_sizes(t, (31, 16, 8, 16, 32)))
        t = torch_npu.npu_format_cast(torch.ones(128, 512, dtype=torch.float16).npu(), 29)
        self.assertTrue(npu_extension.check_storage_sizes(t, (32, 8, 16, 16)))
        # float32 will cast to float16 before calculate
        t = torch_npu.npu_format_cast(torch.ones(128, 512, dtype=torch.float32).npu(), 29)
        self.assertTrue(npu_extension.check_storage_sizes(t, (32, 8, 16, 16)))

    def test_from_blob(self):
        self.assertTrue(npu_extension.check_from_blob())
        self.assertTrue(npu_extension.check_from_blob_strides())

    def test_dispatch_allreduce(self):
        flags = os.O_WRONLY | os.O_RDONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR

        code_file = os.path.join(pathlib.Path(__file__).absolute().parent, "dispatch_allreduce.py")
        log_pth = "allreduce.log"
        with os.fdopen(os.open(log_pth, flags, modes), "w") as f:
            cmd = ["torchrun", "--nproc-per-node=1", code_file]
            p = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=f)
            p.wait()
        
        timeout = 0
        with open(log_pth, 'r', encoding='utf-8') as f:
            tmp = f.readlines()
            for t in tmp:
                print(t)
                if "dispatch timeout" in t:
                    timeout += 1

        os.remove(log_pth)
        self.assertEqual(timeout, 1)

    def test_dump_allreduce(self):
        dump_pth = "./hccl_trace_rank_0"
        code_file = os.path.join(pathlib.Path(__file__).absolute().parent, "dump_allreduce.py")
        cmd = ["torchrun", "--nproc-per-node=1", code_file]
        p = subprocess.Popen(cmd)
        p.wait()

        self.assertTrue(os.path.exists(dump_pth))
        self.assertTrue(os.path.exists(dump_pth + "_py_traceback"))
        os.remove(dump_pth)
        os.remove(dump_pth + "_py_traceback")


if __name__ == "__main__":
    run_tests()
