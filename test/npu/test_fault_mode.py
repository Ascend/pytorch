import os
import subprocess

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
import torch.nn as nn
os.environ["ASCEND_LAUNCH_BLOCKING"] = '0'
import torch_npu


class TestMode(TestCase):

    def test_distributed_launch_param(self):
        process = subprocess.Popen(["python", "-m", "torch.distributed.launch", "--enable-tcmalloc=2", "test.py"],
                                   shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        message = process.stderr.read()
        process.stderr.close()
        process.stdout.close()
        process.terminate()
        process.wait()
        self.assertIn(
            "launch.py: error: unrecognized arguments",
            message
        )

    def test_torchrun_param(self):
        process = subprocess.Popen(["torchrun", "--enable-jemalloc=4", "test.py"], shell=False,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        message = process.stderr.read()
        process.stderr.close()
        process.stdout.close()
        process.terminate()
        process.wait()
        self.assertIn(
            "torchrun: error: unrecognized arguments",
            message
        )

    def test_npu_set_option(self):
        with self.assertRaisesRegex(ValueError, "value of ACL_PRECISION_MODE should be in \\['allow_fp32_to_fp16', "
                                                "'must_keep_origin_dtype'\\] but got allow_fp32_to_fp32"):
            option = {"ACL_PRECISION_MODE": "allow_fp32_to_fp32"}
            torch.npu.set_option(option)

        with self.assertRaisesRegex(TypeError, "npu option must be a dict"):
            torch.npu.set_option("allow_fp32_to_fp32")

        with self.assertRaisesRegex(RuntimeError, "invalid npu option name:ACL_OP_SELECT_IMPL"):
            option = {"ACL_OP_SELECT_IMPL": "high_precision"}
            torch.npu.set_option(option)

    def test_set_device(self):
        with self.assertRaisesRegex(RuntimeError, "Invalid device ID.\n.+Check whether the device ID is valid."):
            torch.npu.set_device(8)

    def test_distributed_init_param(self):
        with self.assertRaisesRegex(ValueError, "Error initializing torch.distributed"):
            torch.distributed.init_process_group(backend="hccl", rank=0, world_size=1)

        with self.assertRaisesRegex(RuntimeError, "Distributed package doesn't have NCCL built in"):
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29689"
            torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

    def test_ckpt_param(self):
        with self.assertRaisesRegex(ValueError,
                                    "Passing `context_fn` or `debug` is only supported when use_reentrant=False"):
            use_reentrant = True
            a = torch.tensor(1., device="npu:0", requires_grad=True)

            def fn(x):
                return x.sin().cos().exp()

            out = checkpoint(fn, a, use_reentrant=use_reentrant, debug=True)
            out.backward()

    def test_load(self):
        path = "./data.pt"
        with self.assertRaisesRegex(FileNotFoundError, "No such file or directory"):
            torch.load(path)

    def test_dataloader(self):
        import torchvision
        from torch.utils.data import DataLoader
        import torchvision.transforms as transforms

        data_transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        with self.assertRaisesRegex(FileNotFoundError, "No such file or directory"):
            train_dataset = torchvision.datasets.ImageFolder(root="imagenet/train", transform=data_transform)
            train_dataset_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    def test_generator(self):
        with self.assertRaisesRegex(RuntimeError, r"Device type CUDA is not supported for torch.Generator\(\) api."):
            torch.Generator(device="cuda")

    def test_not_supported_ops(self):
        command = ['python', '-c', 'import torch; import torch_npu; t = torch.rand(1, 3, 3).npu();t.fmax(t)']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        message = process.stderr.read()
        process.stderr.close()
        process.stdout.close()
        process.terminate()
        process.wait()
        self.assertIn(
            "CAUTION: The operator 'aten::fmax.out' is not currently supported on the NPU backend and will fall back "
            "to run on the CPU. This may have performance implications. (function npu_cpu_fallback)",
            message
        )

    def test_param_verification(self):
        with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device. "
                                                  "Expected NPU tensor, please check whether the input tensor "
                                                  "device is correct."):
            torch.add(torch.rand(2), torch.rand(2).npu())

    def test_custom_ops(self):
        with self.assertRaisesRegex(NotImplementedError, "Could not run 'npu::fast_gelu' with "
                                                         "arguments from the 'CPU' backend."):
            input1 = torch.rand(4)
            torch_npu.fast_gelu(input1)

    def test_autograd_backward(self):
        with self.assertRaisesRegex(RuntimeError, "one of the variables needed for gradient computation has "
                                                  "been modified by an inplace operation"):
            a = torch.randn(5, requires_grad=True)
            d1 = a + 1
            d2 = d1 ** 2
            d1 += 1
            torch.autograd.grad(d2.sum(), a)

    def test_max_memory_allocated(self):
        x = torch.tensor(2).npu()
        with self.assertRaisesRegex(RuntimeError, "Invalid device argument"):
            torch.npu.max_memory_allocated(device="npu:8")

    def test_memory_allocated(self):
        x = torch.tensor(2).npu()
        with self.assertRaisesRegex(RuntimeError, "Invalid device argument"):
            torch.npu.memory_allocated(device="npu:8")

    def test_memory_reserved(self):
        x = torch.tensor(2).npu()
        with self.assertRaisesRegex(RuntimeError, "Invalid device argument"):
            torch.npu.memory_reserved(device="npu:8")

    def test_reset_max_memory_allocated(self):
        x = torch.tensor(2).npu()
        with self.assertRaisesRegex(RuntimeError, "Invalid device argument"):
            torch.npu.reset_max_memory_allocated(device="npu:8")

    def test_aclrtSetDevice(self):
        path = os.path.join(os.path.dirname(__file__), '_fault_mode_cases/error_set_device.py')
        process = subprocess.Popen(["python", f"{path}"], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        message = process.stderr.read()
        process.stderr.close()
        process.stdout.close()
        process.terminate()
        process.wait()
        self.assertIn(
            "_npu_setDevice",
            message
        )
        self.assertIn(
            "Initialize",
            message
        )

    def test_checkpoint_module(self):
        class Net1(nn.Module):
            def __init__(self):
                super().__init__()
                module_list = [
                    nn.Linear(100, 50),
                    nn.ReLU(),
                ]
                self.module_list = nn.ModuleList(module_list)

            def forward(self, input_):
                for layer in self.module_list:
                    input_ = layer(input_)
                return input_

        class Net2(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 6, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(6, 16, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2)
                )

                self.classifiter = nn.Sequential(
                    nn.Linear(16 * 5 * 5, 120),
                    nn.ReLU(),
                    nn.Linear(120, 84),
                    nn.ReLU(),
                    nn.Linear(84, 10)
                )

            def forward(self, input_):
                x = self.features(input_)
                x = x.view(-1, 16 * 5 * 5)
                x = self.classifiter(x)
                return x

        model1 = Net1()
        optimizer1 = torch.optim.SGD(model1.parameters(), 1e-4)

        # resume
        model2 = Net2()
        optimizer2 = torch.optim.SGD(
            [{'params': model2.features.parameters()},
             {'params': model2.classifiter.parameters(), 'lr': 1e-2}], lr=1e-5
        )

        with self.assertRaisesRegex(ValueError, "loaded state dict has a different number of parameter groups"):
            optimizer1.load_state_dict(optimizer2.state_dict())

    def test_aclopCompile(self):
        path = os.path.join(os.path.dirname(__file__), '_fault_mode_cases/error_aclopCompileAndExecute.py')
        process = subprocess.Popen(["python", f"{path}"], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
        out, error = process.communicate(timeout=1800)
        process.stderr.close()
        process.stdout.close()
        process.terminate()
        process.wait()
        self.assertIn(
            "EZ9999",
            error
        )

    def test_ascyn(self):
        path = os.path.join(os.path.dirname(__file__), '_fault_mode_cases/error_aclopCompileAndExecute.py')
        process = subprocess.Popen(["python", f"{path}"], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
        message = process.stderr.read()
        process.stderr.close()
        process.stdout.close()
        process.terminate()
        process.wait()
        self.assertIn(
            "Since the operator is called asynchronously, the stacktrace may be inaccurate. "
            "If you want to get the accurate stacktrace",
            message
        )

    def test_aclrtMallocAlign32(self):
        with self.assertRaisesRegex(RuntimeError, "NPU out of memory. Tried to allocate"):
            x = torch.randn(2000, 2000, 200, 20, device="npu:0")
            y = torch.randn(2000, 2000, 200, 20, device="npu:0")


if __name__ == "__main__":
    run_tests()
