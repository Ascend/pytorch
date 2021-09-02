# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
# All rights reserved.
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

from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import errno
import fcntl
import multiprocessing
import os
import six
import sys
import time
import unittest
from contextlib import contextmanager
from functools import wraps

from itertools import groupby
from functools import reduce
import operator

import torch
import torch.npu
from torch import nn
import torch.nn.functional as F
import torch.distributed as c10d
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch.testing._internal.common_distributed import MultiProcessTestCase
from common_utils import TestCase, run_tests

DEFAULT_TIMEOUT = 100
BACKEND = "hccl"
TEMP_DIR = "/tmp"
INIT_METHOD = "env://"
os.environ['WORLD_SIZE'] = '8'

#change this to your IP
os.environ['MASTER_ADDR'] = 'IP'
os.environ['MASTER_PORT'] = '29501'


SKIP_IF_NO_NPU_EXIT_CODE = 75
SKIP_IF_BACKEND_UNAVAILABLE = 78

def get_timeout():
    return DEFAULT_TIMEOUT


if not dist.is_available():
    print("Distributed not available, skipping tests")
    sys.exit(0)


def skip_if_no_npu_distributed(func):
    """ Hccl multigpu tests requires at least 2 NPUS. Skip if this is not met"""
    func.skip_if_no_npu = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.npu.is_available():
            sys.exit(SKIP_IF_NO_NPU_EXIT_CODE)
        if torch.npu.device_count() < int(os.environ["WORLD_SIZE"]):
            sys.exit(SKIP_IF_NO_NPU_EXIT_CODE)

        return func(*args, **kwargs)

    return wrapper


@contextmanager
def _lock():
    lockfile = os.path.join(TEMP_DIR, "lockfile")
    with open(lockfile, "w") as lf:
        try:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            lf.close()


def _build_tensor(size, value=None):
    if value is None:
        value = size
    return torch.FloatTensor(size, size, size).fill_(value)


class Barrier(object):
    barrier_id = 0

    @classmethod
    def init(cls):
        cls.barrier_id = 0
        barrier_dir = os.path.join(TEMP_DIR, "barrier")
        if not os.path.exists(barrier_dir):
            os.makedirs(barrier_dir)
        for f_name in os.listdir(barrier_dir):
            os.unlink(os.path.join(barrier_dir, f_name))

    @classmethod
    def sync(cls, wait_for=None, timeout=5):
        if wait_for is None:
            wait_for = dist.get_world_size()
        cls.barrier_id += 1
        barrier_dir = os.path.join(TEMP_DIR, "barrier")
        if not os.path.exists(barrier_dir):
            os.makedirs(barrier_dir)
        pid = str(os.getpid())
        barrier_file = os.path.join(barrier_dir, pid)
        with _lock():
            with open(barrier_file, "w") as f:
                f.write(str(cls.barrier_id))

        start_time = time.time()
        while True:
            arrived = 0
            with _lock():
                for f_name in os.listdir(barrier_dir):
                    with open(os.path.join(barrier_dir, f_name), "r") as f:
                        data = f.read()
                        if int(data) >= cls.barrier_id:
                            arrived += 1
            if arrived == wait_for:
                break

            if time.time() - start_time > timeout:
                raise RuntimeError("barrier timeout")
            time.sleep(0.1)


class _DistTestBase(object):
    def _barrier(self, *args, **kwargs):
        Barrier.sync(*args, **kwargs)

    def _init_global_test(self):
        group = list(range(0, dist.get_world_size()))
        group_id = dist.group.WORLD
        rank = dist.get_rank()
        return (group, group_id, rank)

    # HELPER FOR MULTINPU TESTS
    def _init_multinpu_helper(self):
        """Multigpu tests are designed to simulate the multi nodes with multi
        GPUs on each node. Nccl backend requires equal #GPUs in each process.
        On a single node, all visible GPUs are evenly
        divided to subsets, each process only uses a subset.
        """
        nNPUs = torch.npu.device_count()
        world_size = dist.get_world_size()
        visible_devices = range(nNPUs)


        nNPUs_per_process = nNPUs // world_size
        rank_to_NPU = {
            i: list(
                visible_devices[i * nNPUs_per_process: (i + 1) * nNPUs_per_process]
            )
            for i in range(world_size)
        }
        return rank_to_NPU

    # ALL REDUCE
    def _test_all_reduce_helper(
        self,
        group,
        group_id,
        rank,
        op,
        master_value,
        worker_value,
        expected_value,
        rank_to_NPU=None,
    ):
        for src in group:
            if rank == src:
                
                tensor = _build_tensor(src + 1).fill_(master_value)
                device = "npu:" + str(rank_to_NPU[rank][0])
                torch.npu.set_device(device)
                tensor = tensor.to(device)
                
                dist.all_reduce(tensor, op, group_id)

                tensor = tensor.to("cpu")
                self.assertEqual(tensor, _build_tensor(src + 1, expected_value))
            else:
                tensor = _build_tensor(src + 1).fill_(worker_value)
                device = "npu:" + str(rank_to_NPU[rank][0])
                torch.npu.set_device(device)
                tensor = tensor.to(device)
                
                dist.all_reduce(tensor, op, group_id)

                tensor = tensor.to("cpu")
                self.assertEqual(tensor, _build_tensor(src + 1, expected_value))

        self._barrier()

    @skip_if_no_npu_distributed
    def test_all_reduce_sum_npu(self):
        group, group_id, rank = self._init_global_test()
        rank_to_NPU = self._init_multinpu_helper()
        self._test_all_reduce_helper(
            group,
            group_id,
            rank,
            dist.ReduceOp.SUM,
            2,
            10,
            2 + (10 * (len(group) - 1)),
            rank_to_NPU,
        )    


    # BROADCAST
    def _test_broadcast_helper(self, group, group_id, rank, rank_to_NPU=None):

        for ttype, value, requires_npu in [
            ("torch.FloatTensor", -1e-10, False),
            ("torch.DoubleTensor", -1e-100, False),
            ("torch.HalfTensor", -0.1, True),
            ("torch.CharTensor", -2, False),
            ("torch.ByteTensor", 129, False),
            ("torch.IntTensor", -1e5, False),
            ("torch.LongTensor", -1e15, False),
        ]:
            if requires_npu:
                continue
            for src in group:
                expected_tensor = _build_tensor(src + 1, value)
                device = "npu:" + str(rank_to_NPU[rank][0])
                torch.npu.set_device(device)
                expected_tensor = expected_tensor.to(device)
                
                if rank == src:
                    dist.broadcast(expected_tensor, src, group_id)
                else:
                    tensor = _build_tensor(src + 1, -1)
                    device = "npu:" + str(rank_to_NPU[rank][0])
                    torch.npu.set_device(device)
                    tensor = tensor.to(device)
                    dist.broadcast(tensor, src, group_id)

                    tensor = tensor.to("cpu")
                    expected_tensor = expected_tensor.to("cpu")

                    self.assertEqual(tensor.size(), expected_tensor.size())
                    self.assertEqual(tensor.ne(expected_tensor).max(), torch.tensor(False))

        self._barrier()

    @skip_if_no_npu_distributed
    def test_broadcast_npu(self):
        group, group_id, rank = self._init_global_test()
        rank_to_NPU = self._init_multinpu_helper()
        self._test_broadcast_helper(group, group_id, rank, rank_to_NPU)    
    

    # ALL GATHER
    def _test_all_gather_helper(self, group, group_id, rank, rank_to_NPU=None):
        for dest in group:
            #input
            tensor = _build_tensor(dest + 1, rank)
            device = "npu:" + str(rank_to_NPU[rank][0])
            torch.npu.set_device(device)
            tensor = tensor.to(device)
            
            #output
            tensors = [_build_tensor(dest + 1, -1) for i in group]
            new_tensors = []
            for t in tensors:
                torch.npu.set_device(device)
                t = t.to(device)
                new_tensors.append(t) 
            
            #               output_list, input,    group
            dist.all_gather(new_tensors, tensor, group_id)

            #label
            expected_tensors = [_build_tensor(dest + 1, i).to("cpu") for i in group]
            new_tensors = [tensor.to("cpu") for tensor in new_tensors]
            for t1, t2 in zip(new_tensors, expected_tensors):
                self.assertEqual(t1, t2)

        self._barrier()  

    @skip_if_no_npu_distributed
    def test_all_gather_npu(self):
        group, group_id, rank = self._init_global_test()
        rank_to_NPU = self._init_multinpu_helper()
        self._test_all_gather_helper(group, group_id, rank, rank_to_NPU)    


    # REDUCE_SCATTER
    def _test_reduce_scatter_helper(self, group, group_id, rank, op, rank_to_NPU=None):
        for dest in group:
            #output
            tensor = _build_tensor(dest + 1, -1)
            device = "npu:" + str(rank_to_NPU[rank][0])
            torch.npu.set_device(device)
            tensor = tensor.to(device)
            
            #input
            tensors = [_build_tensor(dest + 1, i) for i in group] 
            new_tensors = []
            for t in tensors:
                t= t.to(device)
                new_tensors.append(t)

            #                   output, input_list, op, group, async_op
            dist.reduce_scatter(tensor, new_tensors, op, group_id)

            #label
            expected_tensor = _build_tensor(dest + 1, rank * len(group))

            tensor = tensor.to("cpu")
            self.assertEqual(tensor, expected_tensor)

        self._barrier()

    @skip_if_no_npu_distributed
    def test_reduce_scatter_sum_npu(self):
        group, group_id, rank = self._init_global_test()
        rank_to_NPU = self._init_multinpu_helper()
        self._test_reduce_scatter_helper(group, group_id, rank, dist.ReduceOp.SUM, rank_to_NPU)


    # BARRIER
    def _test_barrier_helper(
            self, group, group_id, rank, rank_to_NPU=None):
        WAIT_TIME = 0.3  # seconds

        for dest in group:
            expected_time = torch.FloatTensor(1).fill_(0.0)
            device = "npu:" + str(rank_to_NPU[rank][0])
            torch.npu.set_device(device)
            expected_time = expected_time.to(device)
            
            if dest == rank:
                expected_time.fill_(time.time() + WAIT_TIME)
                dist.broadcast(expected_time, dest, group_id)
                time.sleep(WAIT_TIME + 0.1)  # sleep a little bit longer
                dist.barrier(group_id)
            else:
                dist.broadcast(expected_time, dest, group_id)
                dist.barrier(group_id)
                self.assertGreaterEqual(
                    float(time.time()),
                    float(expected_time[0]),
                    "destination rank: %d, my rank: %d" % (dest, rank) +
                    " (if you see this failure, please report in #14554)")

        # Use higher timeout for the instance where the test runs
        # against a subgroup and uses a CUDA tensor for expected time.
        # The NPU initialization for the participating processes can
        # take long enough for the barrier timeout to trigger on the
        # process that doesn't participate in the group.
        self._barrier(timeout=20)

    def test_barrier_npu(self):
        group, group_id, rank = self._init_global_test()
        rank_to_NPU = self._init_multinpu_helper()
        self._test_barrier_helper(group, group_id, rank, rank_to_NPU)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class DistributedDataParallelTest(MultiProcessTestCase):
    def setUp(self):
        super(DistributedDataParallelTest, self).setUp()
        self._fork_processes()

    def tearDown(self):
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 2

    def _prepare_single_device_module(self, process_group, devices, device_ids, global_batch_size):
        model = Net()
        torch.npu.set_device(devices[0])

        ddp_model = DistributedDataParallel(
            copy.deepcopy(model).to(devices[0]),
            device_ids=device_ids,
            process_group=process_group,
            bucket_cap_mb=0.001)
        model.to(devices[0])

        input = torch.randn(global_batch_size, 2).to(devices[0])
        target = torch.randn(global_batch_size, 4).to(devices[0])

        return model, ddp_model, input, target

    def _test_ddp_with_process_group(self, process_group, devices, device_ids, multi_device=False):
        local_batch_size = len(devices)
        global_batch_size = self.world_size * local_batch_size

        model, ddp_model, input, target = \
            self._prepare_single_device_module(
                process_group, devices, device_ids, global_batch_size)

        def step_model(model, input, target):
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target.to(output.device))
            loss.backward()

        def update_parameters(model):
            for param in model.parameters():
                with torch.no_grad():
                    param -= param.grad
                param.grad = None

        # Forward
        step_model(model, input, target)
        step_model(ddp_model,
                    input[self.rank * local_batch_size: (self.rank + 1) * local_batch_size],
                    target[self.rank * local_batch_size: (self.rank + 1) * local_batch_size])
        
        # Update weights 
        update_parameters(model)
        update_parameters(ddp_model)
        
        # Check result
        ddp_model = ddp_model.to("cpu")
        model = model.to("cpu")
        self.assertEqual(len(list(model.parameters())), len(list(ddp_model.parameters())))
        for i, j in zip(model.parameters(), ddp_model.parameters()):
            self.assertEqual(i, j)

    def _test_hccl_backend(self, devices, device_ids, multi_device=False):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupHCCL(store, self.rank, self.world_size)
        self._test_ddp_with_process_group(process_group, devices, device_ids, multi_device)

    def test_hccl_backend_whole_train_1step(self):
        npu_rank = [[0],[1],[2],[3],[4],[5],[6],[7]]
        int_devices = npu_rank[self.rank]
        devices = list([torch.device('npu:' + str(i)) for i in int_devices])
        self._test_hccl_backend(devices, int_devices)


class ReducerModule(nn.Module):
    def __init__(self):
        super(ReducerModule, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 4, bias=False)
        self.fc3 = nn.Linear(4, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, use_fc3=True):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        if use_fc3:
            x = self.fc3(x)
        return F.softmax(x, dim=1)


class ReducerTest(TestCase):
    def setUp(self):
        self.store = c10d.FileStore("/dev/null", 1)
        self.process_group = c10d.ProcessGroupHCCL(self.store, 0, 1)

    def _create_single_precision_model(self):
        npu_loc = 'npu:0'
        torch.npu.set_device(npu_loc)
        model = ReducerModule().to(npu_loc)
        return model

    def _create_mixed_precision_model(self):
        npu_loc = 'npu:0'
        model = ReducerModule()
        model.float()
        model.fc1.double()
        return model

    def _create_reducer_for_models(self, models):
        parameters = [list(model.parameters()) for model in models]
        group_by_type = groupby(
            range(len(parameters[0])),
            key=lambda i: parameters[0][i].type())
        buckets = [list(indices) for _, indices in group_by_type]
        return dist.Reducer(parameters, buckets, self.process_group)

    def test_single_dtype_single_bucket(self):
        model = self._create_single_precision_model()
        parameters = list(model.parameters())
        buckets = [list(range(len(parameters)))]
        dist.Reducer([parameters], buckets, self.process_group)

    def test_multi_dtype_single_bucket(self):
        model = self._create_mixed_precision_model()
        # Raise if there are multiple types per bucket.
        # In this case we create one bucket for all parameters.
        with self.assertRaises(RuntimeError):
            parameters = [list(model.parameters())]
            buckets = [list(range(len(parameters[0])))]
            dist.Reducer(parameters, buckets, self.process_group)

    def test_multi_dtype_multi_bucket(self):
        model = self._create_mixed_precision_model()
        parameters = [list(model.parameters())]
        group_by_type = groupby(
            range(len(parameters[0])),
            key=lambda i: parameters[0][i].type())
        buckets = [list(indices) for _, indices in group_by_type]
        dist.Reducer(parameters, buckets, self.process_group)

    def test_forward_backward_unused_parameters(self):
        batch_size = 10
        model = self._create_single_precision_model()
        reducer = self._create_reducer_for_models([model])

        input = torch.rand([batch_size, 2])
        target = torch.rand([batch_size, 4])
        
        npu_loc = "npu:0"
        input = input.to(npu_loc)
        target = target.to(npu_loc)
        
        output = F.mse_loss(model(input, use_fc3=False), target)
        # Check that the grad of fc3 is not set.
        self.assertEqual(None, model.fc3.weight.grad)
        # Compute and accumulate gradients.
        reducer.prepare_for_backward(output)
        output.backward()

        # The reducer will have marked the grad of fc3 as ready, because
        # it doesn't show up in the autograd graph of `output`. Since fc3.weight
        # is considered being globally unused, it will be kept untouched as None.
        self.assertEqual(None, model.fc3.weight.grad)

    def test_forward_backward_optimizer(self):
        batch_size = 10
        model = self._create_single_precision_model()
        reducer = self._create_reducer_for_models([model])
        optimizer = torch.optim.Adam(model.parameters())
        for i in range(3):
            input = torch.rand([batch_size, 2])
            target = torch.rand([batch_size, 4])

            npu_loc = "npu:0"
            input = input.to(npu_loc)
            target = target.to(npu_loc)

            # The `zero_grad` function calls `detach_` and `zero_` on the grad
            # tensors of model parameters. If we tried to set the grad tensors
            # to a view of the reducer's bucket tensors, this would blow up.
            optimizer.zero_grad()

            # Unused parameter only in the first iteration.
            output = F.mse_loss(model(input, use_fc3=(i > 0)), target)
            reducer.prepare_for_backward(output)
            output.backward()
            optimizer.step()


class ComputeBucketAssignmentTest(TestCase):
    def test_single_limit_single_dtype(self):
        tensors = [
            torch.empty([100], dtype=torch.float),
            torch.empty([200], dtype=torch.float),
            torch.empty([100], dtype=torch.float),
            torch.empty([50], dtype=torch.float),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [400])
        self.assertEqual([[0], [1], [2], [3]], result)

    def test_single_limit_multi_dtype(self):
        tensors = [
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [400])
        self.assertEqual([[0, 2], [1, 3], [4], [5]], result)

    def test_multi_limit_single_dtype(self):
        tensors = [
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [40, 80])
        self.assertEqual([[0], [1, 2], [3]], result)

    def test_multi_limit_multi_dtype(self):
        tensors = [
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
        ]
        result = dist._compute_bucket_assignment_by_size(tensors, [200, 400])
        self.assertEqual([[0], [1], [2, 4], [3, 5]], result)


class CommTest(MultiProcessTestCase):
    def setUp(self):
        super(CommTest, self).setUp()
        self._fork_processes()

    def tearDown(self):
        super(CommTest, self).tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def op_timeout_sec(self):
        return 1

    @property
    def world_size(self):
        return 2

    def _test_broadcast_coalesced(self, process_group, device):

        target = torch.arange(60, dtype=torch.float32).chunk(5)
        target += torch.arange(60, dtype=torch.float32).chunk(5)

        # The tensors to pass to broadcast are idential to the target
        # only on the process that is the root of the broadcast.
        if self.rank == 0:
            tensors = list(tensor.clone() for tensor in target)
            torch.npu.set_device(device)
            npu_tensors = []
            for tensor in tensors:
                npu_tensor = tensor.to(device)
                npu_tensors.append(npu_tensor)

        else:
            tensors = list(torch.empty_like(tensor) for tensor in target)
            torch.npu.set_device(device)
            npu_tensors = []
            for tensor in tensors:
                npu_tensor = tensor.to(device)
                npu_tensors.append(npu_tensor)

        c10d._broadcast_coalesced(
            process_group,
            npu_tensors,
            buffer_size=256)

        tensors = []
        for tensor in npu_tensors:
            cpu_tensor = tensor.to("cpu")
            tensors.append(cpu_tensor)

        self.assertEqual(tensors, target)

    def test_broadcast_coalesced_hccl(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupHCCL(store, self.rank, self.world_size)
        device = torch.device('npu:%d' % self.rank)
        self._test_broadcast_coalesced(process_group, device)


if  BACKEND == "hccl":
    WORLD_SIZE = os.environ["WORLD_SIZE"]
    class TestDistBackend(TestCase, _DistTestBase):
        MANAGER_PROCESS_RANK = -1
        TEST_ERROR_EXIT_CODE = 10

        @staticmethod
        def manager_join(fn):
            @wraps(fn)
            def wrapper(self):
                if self.rank == self.MANAGER_PROCESS_RANK:
                    #TODO
                    self._join_and_reduce(fn)
                else:
                    fn(self)

            return wrapper

        @classmethod
        def setUpClass(cls):
            for attr in dir(cls):
                if attr.startswith("test"):
                    fn = getattr(cls, attr)
                    if not getattr(fn, "__unittest_skip__", False):
                        setattr(cls, attr, cls.manager_join(fn))

        def setUp(self):
            super(TestDistBackend, self).setUp()
            self.skip_return_code_checks = []
            self.processes = []
            self.rank = self.MANAGER_PROCESS_RANK
            self.temporary_file = None
            Barrier.init()
            

            for rank in range(int(WORLD_SIZE)):
                self.processes.append(self._spawn_process(rank))
            
        def tearDown(self):
            super(TestDistBackend, self).tearDown()

            # Clean up temporary file if we used one.
            if self.temporary_file:
                try:
                    os.unlink(self.temporary_file.name)
                except OSError as err:
                    # ENOENT is OK because the test is supposed to clean it up.
                    if err.errno != errno.ENOENT:
                        raise

            for p in self.processes:
                p.terminate()

        def _spawn_process(self, rank):
            os.environ["RANK"] = str(rank)
            name = "process " + str(rank)
            # test_distributed.py test suite does not work with spawn
            # mode, so we enforce fork mode for now. In the long term, we should
            # enable spawn mode and refactor this suite to inherit from
            # common_distributed.MultiProcessTestCase.
            if six.PY3:
                # Note: explicitly specifying fork, as spawn is the default in
                # py3.8+ on macos.
                proc_handler = multiprocessing.get_context("fork").Process
            else:
                # fork is the default on Python 2
                proc_handler = multiprocessing.Process
            process = proc_handler(target=self._run, name=name, args=(rank,))
            process.start()
            return process

        def _run(self, rank):
            self.rank = rank
            try:
                dist.init_process_group(
                    init_method=INIT_METHOD,
                    backend=BACKEND,
                    world_size=int(WORLD_SIZE),
                    rank=self.rank
                )
            except RuntimeError as e:
                if "recompile" in e.args[0]:
                    sys.exit(SKIP_IF_BACKEND_UNAVAILABLE)
                    # sys.exit(0)
                raise

            # Execute barrier prior to running test to ensure that every process
            # has finished initialization and that the following test
            # immediately exiting due to a skip doesn't cause flakiness.
            self._barrier()

            getattr(self, self.id().split(".")[2])()
            self._barrier()

            dist.destroy_process_group()
            sys.exit(0)

        def _join_and_reduce(self, fn):
            skip_ok = (
                getattr(fn, "skip_if_no_npu_distributed", False) 
            )
            join_timeout = get_timeout()
            for rank, process in enumerate(self.processes):
                process.join(join_timeout)
                self.assertFalse(
                    process.is_alive(),
                    "Timeout waiting for rank %d to terminate" % rank)

            first_process = self.processes[0]
            for p in self.processes:
                self.assertEqual(p.exitcode, first_process.exitcode)

            if first_process.exitcode == SKIP_IF_BACKEND_UNAVAILABLE:
                raise unittest.SkipTest("Compiled without the " + BACKEND + " backend")

            if skip_ok:
                assert (
                    first_process.exitcode == 0 or
                    first_process.exitcode == SKIP_IF_NO_NPU_EXIT_CODE 
                )

                if first_process.exitcode == SKIP_IF_NO_NPU_EXIT_CODE:
                    raise unittest.SkipTest(
                        "One unique gpu per process is not available"
                    )
            self.assertEqual(first_process.exitcode, 0)
else:
    print("backend is not hccl")

if __name__ == "__main__":
    run_tests()
