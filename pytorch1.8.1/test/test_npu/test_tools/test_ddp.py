# Copyright (c) 2020, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest
import torch
from common_utils import TestCase, run_tests
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    import apex 
    TEST_APEX = True
except ImportError:
    TEST_APEX = False

MASTER_ADDR = '127.0.0.1'
MASTER_PORT = 29456
BACKEND = "hccl"

class ProcessGroupHCCLTest(TestCase):
    def setUp(self):
        super(ProcessGroupHCCLTest, self).setUp()
        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(MASTER_PORT)

    @staticmethod
    def assertequal(data_1, data_2):
        if ((data_1.cpu().float() - data_2.cpu().float()) == 0.).all().item():
            return True 
        else:
            return False

    @staticmethod
    def _init_process_group(rank, WORLD_SIZE):
        dist.init_process_group(backend=BACKEND, world_size=WORLD_SIZE, rank=rank)

    @staticmethod
    def _run_allreduce_op_sum(rank, WORLD_SIZE):
        def allreduce(tensor, op):
            dist.all_reduce(tensor, op)
        
        torch.npu.set_device(rank)
        device = torch.npu.current_device()
        ProcessGroupHCCLTest._init_process_group(rank, WORLD_SIZE)
        input_tensor = torch.tensor([1.,2.,3.,4.]).to(device)
        out = allreduce(input_tensor, dist.ReduceOp.SUM)
        ProcessGroupHCCLTest.assertequal(input_tensor, torch.Tensor([1.,2.,3.,4.])*WORLD_SIZE)

    @staticmethod
    def _run_allreduce_op_product(rank, WORLD_SIZE):
        def allreduce(tensor, op):
            dist.all_reduce(tensor, op)
            return tensor
        
        torch.npu.set_device(rank)
        device = torch.npu.current_device()
        ProcessGroupHCCLTest._init_process_group(rank, WORLD_SIZE)
        input_tensor = torch.tensor([1.,2.,3.,4.]).to(device)
        out = allreduce(input_tensor, dist.ReduceOp.PRODUCT)
        ProcessGroupHCCLTest.assertequal(input_tensor, torch.Tensor([1.,2.,3.,4.])**WORLD_SIZE)

    @staticmethod
    def _run_allreduce_op_max(rank, WORLD_SIZE):
        def allreduce(tensor, op):
            dist.all_reduce(tensor, op)
        
        torch.npu.set_device(rank)
        device = torch.npu.current_device()
        ProcessGroupHCCLTest._init_process_group(rank, WORLD_SIZE)
        input_tensor = torch.tensor([1.,2.,3.,4.]).to(device)*(rank+1)
        out = allreduce(input_tensor, dist.ReduceOp.MAX)
        ProcessGroupHCCLTest.assertequal(input_tensor, torch.Tensor([1.,2.,3.,4.])*WORLD_SIZE)

    @staticmethod
    def _run_allreduce_op_min(rank, WORLD_SIZE):
        def allreduce(tensor, op):
            dist.all_reduce(tensor, op)
        
        torch.npu.set_device(rank)
        device = torch.npu.current_device()
        ProcessGroupHCCLTest._init_process_group(rank, WORLD_SIZE)
        input_tensor = torch.tensor([1.,2.,3.,4.]).to(device)*(rank+1)
        out = allreduce(input_tensor, dist.ReduceOp.MIN)
        ProcessGroupHCCLTest.assertequal(input_tensor, torch.Tensor([2.,4.,6.,8.])*1.)

    def test_all_reduce_op_sum(self):
        WORLD_SIZE = 2
        self.assertTrue(torch.npu.device_count() >= WORLD_SIZE)
        mp.spawn(ProcessGroupHCCLTest._run_allreduce_op_sum, nprocs=WORLD_SIZE, args=(WORLD_SIZE,))

    def test_all_reduce_op_product(self):
        WORLD_SIZE = 2
        self.assertTrue(torch.npu.device_count() >= WORLD_SIZE)
        mp.spawn(ProcessGroupHCCLTest._run_allreduce_op_product, nprocs=WORLD_SIZE, args=(WORLD_SIZE,))
    
    def test_all_reduce_op_max(self):
        WORLD_SIZE = 2
        self.assertTrue(torch.npu.device_count() >= WORLD_SIZE)
        mp.spawn(ProcessGroupHCCLTest._run_allreduce_op_max, nprocs=WORLD_SIZE, args=(WORLD_SIZE,))

    def test_all_reduce_op_min(self):
        WORLD_SIZE = 2
        self.assertTrue(torch.npu.device_count() >= WORLD_SIZE)
        mp.spawn(ProcessGroupHCCLTest._run_allreduce_op_min, nprocs=WORLD_SIZE, args=(WORLD_SIZE,))

    @staticmethod
    def _run_broadcast_op(rank, WORLD_SIZE, src_rank):
        def broadcast(tensor, src_rank):
            dist.broadcast(tensor, src_rank)
        
        torch.npu.set_device(rank)
        device = torch.npu.current_device()
        ProcessGroupHCCLTest._init_process_group(rank, WORLD_SIZE)
        input_tensor = torch.tensor([1.,2.,3.,4.]).to(device)*(rank+1)
        broadcast(input_tensor, src_rank)
        ProcessGroupHCCLTest.assertequal(input_tensor, torch.Tensor([1.,2.,3.,4.])*(src_rank+1))
    
    def test_broadcast_op(self):
        import random
        WORLD_SIZE = 2
        self.assertTrue(torch.npu.device_count() >= WORLD_SIZE)
        src_rank =  random.choice(list(range(WORLD_SIZE)))
        mp.spawn(ProcessGroupHCCLTest._run_broadcast_op, nprocs=WORLD_SIZE, args=(WORLD_SIZE,src_rank))

class SmallModel(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SmallModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channel, out_channel, 3, padding=1)

    def forward(self, input_1):
        input_1 = self.conv1(input_1)
        input_1 = self.relu1(input_1)
        input_1 = self.conv2(input_1)
        return input_1.reshape(input_1.shape[0], -1)

class DistributedDataParallelTest(TestCase):
    input_shape = (4, 3, 24, 24)
    out_shape = (4, 12, 24, 24)

    def setUp(self):
        super(DistributedDataParallelTest, self).setUp()
        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(MASTER_PORT)
    
    @staticmethod
    def _init_process_group(rank, WORLD_SIZE):
        dist.init_process_group(backend=BACKEND, world_size=WORLD_SIZE, rank=rank)

    @staticmethod
    def step_fp32_model(model, input_1, target, criterion, optimizer):
        output = model(input_1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

    @staticmethod
    def step_apex_model(model, input_1, target, criterion, optimizer):
        output = model(input_1)
        loss = criterion(output, target)
        with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.zero_grad()
        optimizer.step()

    @staticmethod
    def _run_single_device_with_fp32(rank, num_device):
        torch.npu.set_device(rank)
        device = torch.npu.current_device()
        DistributedDataParallelTest._init_process_group(rank, num_device)
        model = SmallModel(DistributedDataParallelTest.input_shape[1], 
                            DistributedDataParallelTest.out_shape[1]).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        model_ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        for i in range(10):
            input_1 = torch.rand(DistributedDataParallelTest.input_shape).to(device)
            output_1 =  torch.rand(DistributedDataParallelTest.out_shape).reshape(
                                DistributedDataParallelTest.out_shape[0], -1).to(device)
            DistributedDataParallelTest.step_fp32_model(model_ddp, input_1, output_1, criterion, optimizer)

    @staticmethod
    def _run_single_device_with_apex(rank, num_device):
        torch.npu.set_device(rank)
        device = torch.npu.current_device()
        DistributedDataParallelTest._init_process_group(rank, num_device)
        model = SmallModel(DistributedDataParallelTest.input_shape[1], 
                            DistributedDataParallelTest.out_shape[1]).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O2", loss_scale=1024.)
        model_ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        for i in range(10):
            input_1 = torch.rand(DistributedDataParallelTest.input_shape).to(device)
            output_1 =  torch.rand(DistributedDataParallelTest.out_shape).reshape(
                                DistributedDataParallelTest.out_shape[0], -1).to(device)
            DistributedDataParallelTest.step_apex_model(model_ddp, input_1, output_1, criterion, optimizer)      

    def test_ddp_with_process_group_1_device_fp32(self):
        WORLD_SIZE = 1
        self.assertTrue(torch.npu.device_count() >= WORLD_SIZE)
        mp.spawn(DistributedDataParallelTest._run_single_device_with_fp32, nprocs=WORLD_SIZE, args=(WORLD_SIZE,))

    def test_ddp_with_process_group_multi_device_fp32(self):
        WORLD_SIZE = 2
        self.assertTrue(torch.npu.device_count() >= WORLD_SIZE)
        mp.spawn(DistributedDataParallelTest._run_single_device_with_fp32, nprocs=WORLD_SIZE, args=(WORLD_SIZE,))   
        
    def test_ddp_with_process_group_1_device_apex(self):
        WORLD_SIZE = 1
        self.assertTrue(TEST_APEX)
        mp.spawn(DistributedDataParallelTest._run_single_device_with_apex, nprocs=WORLD_SIZE, args=(WORLD_SIZE,))

    def test_ddp_with_process_group_multi_device_apex(self):
        WORLD_SIZE = 2
        self.assertTrue(torch.npu.device_count() >= WORLD_SIZE)
        self.assertTrue(TEST_APEX)
        mp.spawn(DistributedDataParallelTest._run_single_device_with_apex, nprocs=WORLD_SIZE, args=(WORLD_SIZE,))  

if __name__ == "__main__":
    run_tests()