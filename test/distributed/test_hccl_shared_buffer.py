import unittest
import os
from functools import wraps

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

# Default buffer size is 200
DEFAULT_BUFFER_SIZE = 200


class HcclSharedBufferTest(TestCase):

    @classmethod
    def _init_dist_hccl(cls, rank, world_size, seq):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(seq + 29500)
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        hccl_config1 = {"hccl_buffer_name": "sharedBuffer"}
        options.hccl_config = hccl_config1
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank, pg_options=options)
        return dist

    @classmethod
    # pylint:disable=huawei-too-many-arguments
    def _test_reduce(cls, rank, input1, world_size, init_pg, c2p, p2c, seq):
        #使用共享buffer进行通信，检查最终精度。
        pg = init_pg(rank, world_size, seq)
        input1 = input1.npu()
        pg.all_reduce(input1, op=dist.ReduceOp.SUM, async_op=True)


        ranks = range(world_size)
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        hccl_config = {"hccl_buffer_name": "sharedBuffer"}
        options.hccl_config = hccl_config

        pg2 = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)
        dist.all_reduce(input1, group=pg2, op=dist.ReduceOp.AVG, async_op=True)

        pg = None
        pg2 = None
        c2p.put((rank, input1.cpu(), None))
        p2c.get()

    # pylint:disable=huawei-too-many-arguments
    def _test_multiprocess(self, f, init_pg, expected, input1, world_size, seq):
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(world_size)
        p2c = ctx.Queue(world_size)
        ps = []

        for i in range(world_size):
            p = ctx.Process(
                target=f,
                args=(i, input1.cpu(), world_size, init_pg, c2p, p2c, seq))
            p.start()
            ps.append(p)

        for _ in range(world_size):
            rank, output, mem_diff = c2p.get()
            self.assertEqual(output, expected,
                                "rank {} world_size {} dtype {} shape {} Expect receive tensor {} but got {}.".format(
                                    rank, world_size, expected.dtype, expected.shape, expected, output))
            
            # For case where we want to examine the memory usage of ccl buffer
            if mem_diff:
                for pg in mem_diff:
                    used_mem, expected_mem = mem_diff[pg]
                    self.assertLess(used_mem, expected_mem, 
                                    f"Expected memory used to be less than {expected_mem} for {pg}, but got {used_mem}")

        for _ in range(world_size):
            p2c.put(0)

        for p in ps:
            p.join()

    def _construct_excepted_result(self, inputs, world_size, dtype=np.float32):
        # dist.ReduceOp.SUM
        op1_expected = 0
        for _ in range(world_size):
            op1_expected += inputs
        
        #dist.ReduceOp.AVG
        expected = 0
        for _ in range(world_size):
            expected += op1_expected
        expected = expected / world_size
        return expected

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_dist(self):
        ranks = [2]
        dtype_list = [np.float32, np.float16]
        format_list = [3, 29]
        shape_format = [
            [i, j, [12, 56, 256]]
            for i in dtype_list
            for j in format_list
        ]
        seq = 0
        for world_size in ranks:
            if torch.npu.device_count() < world_size:
                continue
            for shape in shape_format:
                if shape[0] == np.int8:
                    shape[1] = 0
                exp_input, input1 = create_common_tensor(shape, -10, 10)
                expected = self._construct_excepted_result(exp_input, world_size)
                seq = seq + 1
                self._test_multiprocess(HcclSharedBufferTest._test_reduce,
                                        HcclSharedBufferTest._init_dist_hccl, expected, input1, world_size, seq)

    @classmethod
    # pylint:disable=huawei-too-many-arguments
    def _test_buffer_name(cls, rank, input1, world_size, init_pg, c2p, p2c, seq):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(seq + 29400)
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch.npu.set_device(rank)
        input1 = input1.npu()
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()

        hccl_config1 = {"hccl_buffer_name": "shared0"} # 创建一个共享buffer
        options.hccl_config = hccl_config1
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank, pg_options=options)
        dist.all_reduce(input1, async_op=True)

        hccl_config2 = {"hccl_buffer_name": "shared2"} # 注释1：只在创建pg的时候指定参数。后续修改不影响创建的名字,当前pg保持创建时的名字。
        options.hccl_config = hccl_config2
        dist.all_reduce(input1, async_op=True)

        ranks = range(world_size)
        hccl_config3 = {"hccl_buffer_name": "shared0"}
        options.hccl_config = hccl_config3

        pg = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)
        dist.all_reduce(input1, group=pg, async_op=True)

        hccl_config4 = {"hccl_buffer_name": "shared4"} # 同注释1。
        options.hccl_config = hccl_config4
        dist.all_reduce(input1, group=pg, async_op=True)

        hccl_config5 = {"hccl_buffer_name": "shared5"}
        options.hccl_config = hccl_config5
        pg2 = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)
        dist.all_reduce(input1, group=pg2, async_op=True)

        hccl_config6 = {"hccl_buffer_name": "shared6"} # 同注释1。
        options.hccl_config = hccl_config6
        dist.all_reduce(input1, group=pg2, async_op=True)

        hccl_config20 = {}
        options.hccl_config = hccl_config20
        pg20 = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)
        dist.all_reduce(input1, group=pg20, async_op=True)

        output = input1
        output.fill_(0)
        c2p.put((rank, output.cpu(), None))
        p2c.get()

    @skipIfUnsupportMultiNPU(2)
    def test_sharedbuffer_dist(self):
        ranks = [2]
        dtype_list = [np.uint8]
        format_list = [2]
        shape_format = [
                           [i, j, [12, 56, 256]] for i in dtype_list for j in format_list
                       ] + [[i, j, [1]] for i in dtype_list for j in format_list]
        seq = 0
        for world_size in ranks:
            for shape in shape_format:
                if len(shape[2]) == 1:
                    continue
                exp_input, input1 = create_common_tensor(shape, -10, 10)
                expected = exp_input.fill_(0)
                seq = seq + 1
                self._test_multiprocess(HcclSharedBufferTest._test_buffer_name,
                                        None, expected, input1, world_size, seq)

    @classmethod
    # pylint:disable=huawei-too-many-arguments
    def _test_buffer_refcount(cls, rank, input1, world_size, init_pg, c2p, p2c, seq):
        print(f"rank table pid = {os.getpid()},rank = {rank}, world_size = {world_size}")
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(seq + 29600)
        os.environ['RANK_TABLE_FILE'] = '/data/syg/source/pytorch_npu/test/distributed/ranktable.json'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch.npu.set_device(rank)
        input1 = input1.npu()
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        hccl_config = {"hccl_buffer_name": "sharedBuffer"} # 创建一个共享buffer
        options.hccl_config = hccl_config
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank, pg_options=options)

        ranks = range(world_size)
        hccl_config1 = {"hccl_buffer_name": "subSharedBuffer"}
        options.hccl_config = hccl_config1
        options.global_ranks_in_group = ranks
        pg1 = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)
        dist.all_reduce(input1, group=pg1, async_op=True)

        dist.all_reduce(input1, async_op=True)

        hccl_config2 = {"hccl_buffer_name": "sharedBuffer"}
        options.hccl_config = hccl_config2
        pg2 = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)
        dist.all_reduce(input1, group=pg2, async_op=True)

        torch_npu.npu.synchronize()
        del pg1
        del pg2

        hccl_config3 = {"hccl_buffer_name": "subSharedBuffer"} # 
        options.hccl_config = hccl_config3
        pg3 = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)
        dist.all_reduce(input1, group=pg3, async_op=True)
        pg3 = None

        output = input1
        output.fill_(0)
        c2p.put((rank, output.cpu(), None))
        p2c.get()

    @skipIfUnsupportMultiNPU(2)
    def test_reduce_refcount_dist(self):
        ranks = [2]
        dtype_list = [np.uint8]
        format_list = [2]
        shape_format = [
                           [i, j, [12, 56, 256]] for i in dtype_list for j in format_list
                       ] + [[i, j, [1]] for i in dtype_list for j in format_list]
        seq = 0
        for world_size in ranks:
            for shape in shape_format:
                if len(shape[2]) == 1:
                    continue
                exp_input, input1 = create_common_tensor(shape, -10, 10)
                expected = exp_input.fill_(0)
                seq = seq + 1
                self._test_multiprocess(HcclSharedBufferTest._test_buffer_refcount,
                                        None, expected, input1, world_size, seq)

    @staticmethod
    def get_memory_info():
        device_id = torch.npu.current_device()
        reserved = torch.npu.memory_reserved(device_id)
        max_allocated = torch.npu.max_memory_allocated(device_id)
        pta_reserved_MB = reserved / 1024 / 1024
        pta_activated_MB = max_allocated / 1024 / 1024
        memory_info_after = torch.npu.mem_get_info(device_id)
        return pta_reserved_MB, pta_activated_MB, memory_info_after

    @staticmethod
    def reset_memory_info():
        device_id = torch.npu.current_device()
        torch.npu.reset_peak_memory_stats()
        torch.npu.reset_max_memory_allocated()
        mem_info_start = torch.npu.mem_get_info(device_id)
        return mem_info_start

    @classmethod
    # pylint:disable=huawei-too-many-arguments
    def _test_buffer_memory(cls, rank, input1, world_size, init_pg, c2p, p2c, seq):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(seq + 29600)
        os.environ['RANK_TABLE_FILE'] = '/data/syg/source/pytorch_npu/test/distributed/ranktable.json'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        hccl_buffsize = os.environ.get('HCCL_BUFFSIZE')

        # set buffer size per group
        if hccl_buffsize:
            buffer_size = int(hccl_buffsize) * 2
        else:
            buffer_size = DEFAULT_BUFFER_SIZE * 2

        # set a small tol for inputs and other memory use, make sure its less than a normal buffer size
        tol = buffer_size * 0.25

        num_different_buffer_group = 0
        torch.npu.set_device(rank)
        input1 = input1.npu()
        mem_diff = {}
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        hccl_config = {"hccl_buffer_name": "newBuffer"}  # global pg options
        options.hccl_config = hccl_config

        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank, pg_options=options)

        # reset the memory stats
        memory_info_before = cls.reset_memory_info()

        # create pg1 with a buffer
        ranks = range(world_size)
        hccl_config1 = {"hccl_buffer_name": "subSharedBuffer"}
        options.hccl_config = hccl_config1
        options.global_ranks_in_group = ranks
        pg1 = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)
        dist.all_reduce(input1, group=pg1, async_op=True)

        # get pg1 mem info
        _, _, mem_after = cls.get_memory_info()
        diff = (memory_info_before[0] - mem_after[0]) / 1024 / 1024
        num_different_buffer_group += 1
        expected = buffer_size * num_different_buffer_group + tol
        mem_diff["pg1"] = [diff, expected]

        # create pg2 with the same buffer
        hccl_config2 = {"hccl_buffer_name": "subSharedBuffer"}
        options.hccl_config = hccl_config2
        pg2 = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)
        dist.all_reduce(input1, group=pg2, async_op=True)

        # get pg2 mem info
        _, _, mem_after = cls.get_memory_info()
        diff = (memory_info_before[0] - mem_after[0]) / 1024 / 1024
        expected = buffer_size * num_different_buffer_group + tol
        mem_diff["pg2"] = [diff, expected]

        # create pg3 with a different buffer
        hccl_config3 = {"hccl_buffer_name": "subSharedBuffer_2"}
        options.hccl_config = hccl_config3
        pg3 = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)
        dist.all_reduce(input1, group=pg3, async_op=True)

        # get pg3 mem info
        _, _, mem_after = cls.get_memory_info()
        diff = (memory_info_before[0] - mem_after[0]) / 1024 / 1024
        num_different_buffer_group += 1
        expected = buffer_size * num_different_buffer_group + tol
        mem_diff["pg3"] = [diff, expected]
        torch_npu.npu.synchronize()

        output = input1
        output.fill_(0)
        c2p.put((rank, output.cpu(), mem_diff))
        p2c.get()

        return True
    
    @classmethod
    # pylint:disable=huawei-too-many-arguments
    def _test_buffer_memory_with_deleted_pg(cls, rank, input1, world_size, init_pg, c2p, p2c, seq):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(seq + 29600)
        os.environ['RANK_TABLE_FILE'] = '/data/syg/source/pytorch_npu/test/distributed/ranktable.json'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        hccl_buffsize = os.environ.get('HCCL_BUFFSIZE')

        # set buffer size per group
        if hccl_buffsize:
            buffer_size = int(hccl_buffsize) * 2
        else:
            buffer_size = DEFAULT_BUFFER_SIZE * 2

        # set a small tol for inputs and other memory use, make sure its less than a normal buffer size
        tol = buffer_size * 0.25

        num_different_buffer_group = 0
        torch.npu.set_device(rank)
        input1 = input1.npu()
        mem_diff = {}
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        hccl_config = {"hccl_buffer_name": "newBuffer"}  # global pg options
        options.hccl_config = hccl_config

        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank, pg_options=options)

        # reset the memory stats
        memory_info_before = cls.reset_memory_info()

        # create pg1 with a buffer
        ranks = range(world_size)
        hccl_config1 = {"hccl_buffer_name": "subSharedBuffer"}
        options.hccl_config = hccl_config1
        options.global_ranks_in_group = ranks
        pg1 = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)
        dist.all_reduce(input1, group=pg1, async_op=True)

        # get pg1 mem info
        _, _, mem_after = cls.get_memory_info()
        diff = (memory_info_before[0] - mem_after[0]) / 1024 / 1024
        num_different_buffer_group += 1
        expected = buffer_size * num_different_buffer_group + tol
        mem_diff["pg1"] = [diff, expected]

        # create pg2 with the same buffer
        hccl_config2 = {"hccl_buffer_name": "subSharedBuffer"}
        options.hccl_config = hccl_config2
        pg2 = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)
        dist.all_reduce(input1, group=pg2, async_op=True)

        # get pg2 mem info
        _, _, mem_after = cls.get_memory_info()
        diff = (memory_info_before[0] - mem_after[0]) / 1024 / 1024
        expected = buffer_size * num_different_buffer_group + tol
        mem_diff["pg2"] = [diff, expected]

        # delete the first two pg
        torch_npu.npu.synchronize()
        del pg1
        del pg2

        # create pg3 with a different buffer
        memory_info_before = cls.reset_memory_info()
        hccl_config3 = {"hccl_buffer_name": "subSharedBuffer2"}
        options.hccl_config = hccl_config3
        pg3 = dist.new_group(backend='hccl', ranks=ranks, pg_options=options)
        dist.all_reduce(input1, group=pg3, async_op=True)

        # get pg3 mem info
        _, _, mem_after = cls.get_memory_info()
        diff = (memory_info_before[0] - mem_after[0]) / 1024 / 1024
        num_different_buffer_group = 1
        expected = buffer_size * num_different_buffer_group + tol
        mem_diff["pg3"] = [diff, expected]

        output = input1
        output.fill_(0)
        c2p.put((rank, output.cpu(), mem_diff))
        p2c.get()

        return True

    @skipIfUnsupportMultiNPU(2)
    def test_memcheck(self):
        ranks = [2]
        dtype_list = [np.uint8]
        format_list = [2]
        shape_format = [
                        [i, j, [12, 56, 256]] for i in dtype_list for j in format_list
                    ] + [[i, j, [1]] for i in dtype_list for j in format_list]
        seq = 0
        for world_size in ranks:
            for shape in shape_format:
                if len(shape[2]) == 1:
                    continue
                exp_input, input1 = create_common_tensor(shape, -10, 10)
                expected = exp_input.fill_(0)
                seq = seq + 1
                self._test_multiprocess(HcclSharedBufferTest._test_buffer_memory,
                                        None, expected, input1, world_size, seq)
                self._test_multiprocess(HcclSharedBufferTest._test_buffer_memory_with_deleted_pg,
                                        None, expected, input1, world_size, seq)

if __name__ == '__main__':
    run_tests()