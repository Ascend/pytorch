import os
import unittest
import numpy as np

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.npu.utils import get_cann_version, _is_gte_cann_version

RESERVED_BYTES_CURRENT = "reserved_bytes.current"
ALLOCATED_BYTES_CURRENT = "allocated_bytes.current"


def get_skip():
    try:
        if not _is_gte_cann_version("8.5.0", "CANN"):
            return True

        driver_version = get_cann_version(module="DRIVER")
        parts = [int(x) for x in driver_version.split('.') if x.isdigit()]
        if len(parts) < 2:
            return True

        major = parts[0]
        minor = parts[1]
        if major < 25:
            return True

        if major > 25:
            return False

        return minor < 4
    except Exception:
        return True


skip = get_skip()


@unittest.skipIf(skip, "cann or driver version not supported")
class TestHostCachingAllocator(TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'pin_memory_expandable_segments:True'

    
    def test_allocate_with_block_cut(self):
        # 申请一个64M的tensor, 会预选绑定80M物理内存
        memory_64m = torch.ones([1024, 1024, 16]).pin_memory()
        # 实际分配的内存是申请大小 = 512 * (size + 32 + 512 - 1) / 512
        self.assertEqual(torch_npu.npu.host_memory_stats().get(ALLOCATED_BYTES_CURRENT, 0), 64 * 1024 * 1024 + 512)
        self.assertEqual(torch_npu.npu.host_memory_stats().get(RESERVED_BYTES_CURRENT, 0), 80 * 1024 * 1024)

        # 释放64M内存，内存归还内存池，物理内存不变，使用量为0
        memory_64m = None
        self.assertEqual(torch_npu.npu.host_memory_stats().get(RESERVED_BYTES_CURRENT, 0), 80 * 1024 * 1024)
        self.assertEqual(torch_npu.npu.host_memory_stats().get(ALLOCATED_BYTES_CURRENT, 0), 0)
        # 再次申请4M/8M/16M大小的内存后，总的预留内存应该不变
        memory_4m = torch.ones([1024, 1024, 1]).pin_memory()
        memory_8m = torch.ones([1024, 1024, 2]).pin_memory()
        memory_16m = torch.ones([1024, 1024, 4]).pin_memory()
        self.assertEqual(torch_npu.npu.host_memory_stats().get(ALLOCATED_BYTES_CURRENT, 0), (4 + 8 + 16) * 1024 * 1024 + 512 * 3)
        self.assertEqual(torch_npu.npu.host_memory_stats().get(RESERVED_BYTES_CURRENT, 0), 80 * 1024 * 1024)
        # 内存块归还内存池, 执行host_empty_cache释放所有物理内存防止用例间相互影响
        memory_4m = None
        memory_8m = None
        memory_16m = None
        torch_npu.npu.host_empty_cache()
        self.assertEqual(torch_npu.npu.host_memory_stats().get(RESERVED_BYTES_CURRENT, 0), 0)
        self.assertEqual(torch_npu.npu.host_memory_stats().get(ALLOCATED_BYTES_CURRENT, 0), 0)


    def test_allocate_with_block_merge(self):
        # 先申请两个内存空间为40M的内存释放掉，实际分配的物理内存应该为100M
        memory_40m1 = torch.ones([1024, 1024, 10]).pin_memory()
        memory_40m2 = torch.ones([1024, 1024, 10]).pin_memory()
        memory_40m1 = None
        memory_40m2 = None
        self.assertEqual(torch_npu.npu.host_memory_stats().get(RESERVED_BYTES_CURRENT, 0), 100 * 1024 * 1024)
        # 再申请一个64M的tensor, 不会再申请物理内存
        memory_64m = torch.ones([1024, 1024, 16], pin_memory=True).pin_memory()
        self.assertEqual(torch_npu.npu.host_memory_stats().get(RESERVED_BYTES_CURRENT, 0), 100 * 1024 * 1024)
        self.assertEqual(torch_npu.npu.host_memory_stats().get(ALLOCATED_BYTES_CURRENT, 0), 64 * 1024 * 1024 + 512)
        # 释放空闲block的物理内存, 应该只能释放20M，已分配内存不变
        torch_npu.npu.host_empty_cache()
        self.assertEqual(torch_npu.npu.host_memory_stats().get(ALLOCATED_BYTES_CURRENT, 0), 64 * 1024 * 1024 + 512)
        self.assertEqual(torch_npu.npu.host_memory_stats().get(RESERVED_BYTES_CURRENT, 0), 80 * 1024 * 1024)
        # 释放所有内存防止用例间相互影响
        memory_64m = None
        torch_npu.npu.host_empty_cache()
        self.assertEqual(torch_npu.npu.host_memory_stats().get(ALLOCATED_BYTES_CURRENT, 0), 0)
        self.assertEqual(torch_npu.npu.host_memory_stats().get(RESERVED_BYTES_CURRENT, 0), 0)


    def test_h2d_inplace(self):
        def cpu_copy_op_exec(input1_host, input2_host):
            input1_host.add_(input1_host)
            input1_host.copy_(input2_host)
            return input1_host.numpy()

        def npu_copy_op_exec(input1_device, input2_host):
            input1_device.add_(input1_device)
            input1_device.copy_(input2_host)
            return input1_device.to("cpu").numpy()

        cpu_input1, npu_input1 = create_common_tensor([np.float32, -1, (5, 5)], 1, 100)
        cpu_input2 = cpu_input1 + 1
        cpu_output = cpu_copy_op_exec(cpu_input1, cpu_input2)
        npu_output = npu_copy_op_exec(npu_input1, cpu_input2)
        self.assertRtolEqual(cpu_output, npu_output)


    def test_d2h_inplace(self):
        def cpu_copy_op_exec(input1_host, input2_host):
            input1_host.add_(input1_host)
            input2_host.copy_(input1_host)
            return input2_host.numpy()

        def npu_copy_op_exec(input1_device, input2_host):
            input1_device.add_(input1_device)
            input2_host.copy_(input1_device)
            return input2_host.numpy()

        cpu_input1, npu_input1 = create_common_tensor([np.float32, -1, (5, 5)], 1, 100)
        cpu_out1 = cpu_input1 + 1
        cpu_out2 = cpu_input1 + 1
        cpu_output = cpu_copy_op_exec(cpu_input1, cpu_out1)
        npu_output = npu_copy_op_exec(npu_input1, cpu_out2)
        self.assertRtolEqual(cpu_output, npu_output)

    
    def test_event_free(self):
        tensor = torch.ones([1024, 1024, 16]).npu()
        tensor_cpu = torch.ones([1024, 1024, 16]).pin_memory()
        tensor_cpu.copy_(tensor, non_blocking=True)
        tensor_cpu = None

        torch_npu.npu.host_empty_cache()
        self.assertEqual(torch_npu.npu.host_memory_stats().get(ALLOCATED_BYTES_CURRENT, 0), 64 * 1024 * 1024 + 512)
        self.assertEqual(torch_npu.npu.host_memory_stats().get(RESERVED_BYTES_CURRENT, 0), 80 * 1024 * 1024)

        torch.npu.synchronize()
        torch_npu.npu.host_empty_cache()
        self.assertEqual(torch_npu.npu.host_memory_stats().get(ALLOCATED_BYTES_CURRENT, 0), 0)
        self.assertEqual(torch_npu.npu.host_memory_stats().get(RESERVED_BYTES_CURRENT, 0), 0)


    def test_muti_stream(self):
        device = torch.device("npu:0")
        stream1 = torch.npu.Stream(device=device)
        stream2 = torch.npu.Stream(device=device)

        for i in range(1, 5):
            tensor = torch.ones([100, 100, i * 10], device=device)
            event1 = torch.npu.Event()

            with torch.npu.stream(stream1):
                tensor1 = tensor + 10
                event1.record()
                tensor1_cpu = torch.ones([100, 100, i * 10]).pin_memory()
                tensor1_cpu.copy_(tensor1, non_blocking=True)

            with torch.npu.stream(stream2):
                stream2.wait_event(event1)
                tensor2 = tensor1 * 2
                tensor2_cpu = torch.ones([100, 100, i * 10]).pin_memory()
                tensor2_cpu.copy_(tensor2, non_blocking=True)

            torch.npu.synchronize()
            self.assertRtolEqual(tensor2_cpu.numpy(), (torch.ones([100, 100, i * 10]).cpu() * 22).numpy())
            tensor1_cpu = None
            tensor2_cpu = None
            tensor = None
            tensor1 = None
            tensor2 = None

        torch.npu.synchronize()
        torch_npu.npu.host_empty_cache()
        self.assertEqual(torch_npu.npu.host_memory_stats().get(ALLOCATED_BYTES_CURRENT, 0), 0)
        self.assertEqual(torch_npu.npu.host_memory_stats().get(RESERVED_BYTES_CURRENT, 0), 0)


    @classmethod
    def tearDownClass(cls):
        instance = cls()
        instance.assertEqual(torch_npu.npu.host_memory_stats().get("allocated_bytes.peak", 0), 80 * 1024 * 1024 + 512 * 2)
        instance.assertEqual(torch_npu.npu.host_memory_stats().get(ALLOCATED_BYTES_CURRENT, 0), 0)
        instance.assertEqual(torch_npu.npu.host_memory_stats().get("reserved_bytes.peak", 0), 100 * 1024 * 1024)
        instance.assertEqual(torch_npu.npu.host_memory_stats().get(RESERVED_BYTES_CURRENT, 0), 0)


if __name__ == '__main__':
    run_tests()