import os
import stat
import pathlib
import subprocess
import unittest
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.common_utils import create_common_tensor

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

    def test_op_hook_with_add(self):
        # init
        input_1 = torch.tensor((4, 4))
        input_2 = torch.tensor((4, 4))
        expected = torch.add(input_1, input_2)
        input_1 = input_1.npu()
        input_2 = input_2.npu()
        npu_extension.reset_op_hook_call_count()
        count = npu_extension.get_op_hook_call_count()
        self.assertEqual(count, 0)

        # register op_hook, but not enable it
        npu_extension.register_op_hook()
        output_1 = torch.add(input_1, input_2)
        count = npu_extension.get_op_hook_call_count()
        self.assertEqual(count, 0)

        # enable op_hook
        torch.npu.set_option({"OP_HOOK_ENABLE": "enable"})
        output_2 = torch.add(input_1, input_2)
        count = npu_extension.get_op_hook_call_count()
        self.assertEqual(count, 5)

        # disable op_hook
        torch.npu.set_option({"OP_HOOK_ENABLE": "disable"})
        output_3 = torch.add(input_1, input_2)
        count = npu_extension.get_op_hook_call_count()
        self.assertEqual(count, 5)

        # final
        self.assertEqual(output_1.cpu(), expected)
        self.assertEqual(output_2.cpu(), expected)
        self.assertEqual(output_3.cpu(), expected)

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_op_hook_with_all_reduce(cls, rank, input1, world_size, init_pg, c2p):
        # init
        dist_group = init_pg(rank, world_size)
        dst = 0
        input_1 = input1.npu()
        input_2 = input1.npu()
        input_3 = input1.npu()
        npu_extension.reset_op_hook_call_count()
        count_1 = npu_extension.get_op_hook_call_count()

        # register op_hook, but not enable it
        npu_extension.register_op_hook()
        dist_group.all_reduce(input_1)
        count_2 = npu_extension.get_op_hook_call_count()

        # enable op_hook
        torch.npu.set_option({"OP_HOOK_ENABLE": "enable"})
        dist_group.all_reduce(input_2)
        count_3 = npu_extension.get_op_hook_call_count()

        # disable op_hook
        torch.npu.set_option({"OP_HOOK_ENABLE": "disable"})
        dist_group.all_reduce(input_3)
        count_4 = npu_extension.get_op_hook_call_count()

        # final
        all_reduce_ouput = (input_1.cpu(), input_2.cpu(), input_3.cpu())
        op_hook_count = (count_1, count_2, count_3, count_4)
        c2p.put((rank, dst, all_reduce_ouput, op_hook_count))

    def _test_multiprocess(self, f, init_pg, expected, input1, world_size):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)

        ps = []
        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, input1.cpu(), world_size, init_pg, c2p))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, dst, all_reduce_ouput, op_hook_count = c2p.get()
            output_1, output_2, output_3 = all_reduce_ouput
            count_1, count_2, count_3, count_4 = op_hook_count
            if rank == dst:
                self.assertEqual(count_1, 0)
                self.assertEqual(count_2, 0)
                self.assertEqual(count_3, 3)
                self.assertEqual(count_4, 3)
                self.assertEqual(output_1, expected,
                                 ("rank {} Expect receive tensor {} but got {}.").format(rank, expected, output_1))
                self.assertEqual(output_2, expected,
                                 ("rank {} Expect receive tensor {} but got {}.").format(rank, expected, output_2))
                self.assertEqual(output_3, expected,
                                 ("rank {} Expect receive tensor {} but got {}.").format(rank, expected, output_3))

        for p in ps:
            p.join()

    @skipIfUnsupportMultiNPU(2)
    def test_op_hook_with_all_reduce(self):
        # CI currently supports only 2 devices
        ranks = [2]
        shape = [np.float32, 2, [2, 3, 16]]
        for world_size in ranks:
            exp_input, input1 = create_common_tensor(shape, -10, 10)
            expected = 0
            for _ in range(world_size):
                expected += exp_input
            self._test_multiprocess(TestCppExtensionAOT._test_op_hook_with_all_reduce,
                                    TestCppExtensionAOT._init_dist_hccl, expected, input1, world_size)

if __name__ == "__main__":
    run_tests()
