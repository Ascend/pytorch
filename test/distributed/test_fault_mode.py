import os.path
import subprocess
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestMode(TestCase):

    @skipIfUnsupportMultiNPU(2)
    def test_dist_init_duplicate(self):
        path = os.path.join(os.path.dirname(__file__), '_fault_mode_cases/error_double_init.py')
        process = subprocess.Popen(["torchrun", "--nproc-per-node=2", f"{path}"], shell=False, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        message = process.stderr.read()
        process.stderr.close()
        process.stdout.close()
        process.terminate()
        process.wait()
        self.assertIn(
            "trying to initialize the default process group twice!",
            message
        )

    @skipIfUnsupportMultiNPU(2)
    def test_dist_use_same_addr(self):
        path = os.path.join(os.path.dirname(__file__), '_fault_mode_cases/error_use_same_addr.py')

        processes = []
        p = subprocess.Popen(["torchrun", "--nproc-per-node=2", f"{path}", "&"], shell=False, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
        processes.append(p)
        p = subprocess.Popen(["torchrun", "--nproc-per-node=2", f"{path}"], shell=False, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
        processes.append(p)
        error_info = ""
        for index, p in enumerate(processes):
            stdout, stderr = p.communicate(timeout=1800)
            error_info = error_info + " " + stderr

        self.assertIn(
            "address already in use",
            error_info
        )

    @skipIfUnsupportMultiNPU(2)
    def test_broadcast_group(self):
        path = os.path.join(os.path.dirname(__file__), '_fault_mode_cases/error_broadcast_param.py')
        process = subprocess.Popen(["torchrun", "--nproc-per-node=2", f"{path}"], shell=False, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        message = process.stderr.read()
        process.stderr.close()
        process.stdout.close()
        process.terminate()
        process.wait()
        self.assertIn(
            "UserWarning: Running broadcast on global rank 1 which does not belong to the given group.",
            message
        )

    @skipIfUnsupportMultiNPU(2)
    def test_cpu_tensor(self):
        path = os.path.join(os.path.dirname(__file__), '_fault_mode_cases/error_cpu_tensor.py')
        process = subprocess.Popen(["torchrun", "--nproc-per-node=2", f"{path}"], shell=False, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        message = process.stderr.read()
        process.stderr.close()
        process.stdout.close()
        process.terminate()
        process.wait()
        self.assertIn(
            "No backend type associated with device type cpu",
            message
        )

    @skipIfUnsupportMultiNPU(2)
    def test_diff_dtype(self):
        path = os.path.join(os.path.dirname(__file__), '_fault_mode_cases/error_diff_dtype.py')
        process = subprocess.Popen(["torchrun", "--nproc-per-node=2", f"{path}"], shell=False, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        message = process.stderr.read()
        process.stderr.close()
        process.stdout.close()
        process.terminate()
        process.wait()
        self.assertIn(
            "Invalid usage of tensors with different dtypesFound torch.float32 and  torch.int64",
            message
        )

    @skipIfUnsupportMultiNPU(2)
    def test_diff_type(self):
        path = os.path.join(os.path.dirname(__file__), '_fault_mode_cases/error_diff_type.py')
        process = subprocess.Popen(["torchrun", "--nproc-per-node=2", f"{path}"], shell=False, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        message = process.stderr.read()
        process.stderr.close()
        process.stdout.close()
        process.terminate()
        process.wait()
        self.assertIn(
            "Invalid function argument. Expected parameter `output` of type torch.Tensor",
            message
        )

    @skipIfUnsupportMultiNPU(2)
    def test_error_size(self):
        path = os.path.join(os.path.dirname(__file__), '_fault_mode_cases/error_size.py')
        process = subprocess.Popen(["torchrun", "--nproc-per-node=2", f"{path}"], shell=False, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        message = process.stderr.read()
        process.stderr.close()
        process.stdout.close()
        process.terminate()
        process.wait()
        self.assertIn(
            "Tensor list input to scatter/gather must match number of collective participants",
            message
        )

    @skipIfUnsupportMultiNPU(2)
    def test_discontinuous_tensor(self):
        path = os.path.join(os.path.dirname(__file__), '_fault_mode_cases/error_discontinuous_tensor.py')
        process = subprocess.Popen(["torchrun", "--nproc-per-node=2", f"{path}"], shell=False, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        message = process.stderr.read()
        process.stderr.close()
        process.stdout.close()
        process.terminate()
        process.wait()
        self.assertIn(
            "RuntimeError: Tensors must be contiguous",
            message
        )

    @skipIfUnsupportMultiNPU(2)
    def test_hccl_timeout(self):
        path = os.path.join(os.path.dirname(__file__), '_fault_mode_cases/error_hccl_timeout.py')
        process = subprocess.Popen(["torchrun", "--nproc-per-node=2", f"{path}"], shell=False, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        message = process.stderr.read()
        process.stderr.close()
        process.stdout.close()
        process.terminate()
        process.wait()
        self.assertIn(
            "EI0002",
            message
        )



if __name__ == "__main__":
    run_tests()
