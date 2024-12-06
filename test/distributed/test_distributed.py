from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import fcntl
import os
import sys
import time
import tempfile
import unittest
import logging
import traceback
from contextlib import contextmanager
from datetime import timedelta
from functools import reduce, wraps
import types
from collections import namedtuple
from multiprocessing import Manager

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT

import torch_npu
from torch_npu.utils._path_manager import PathManager
from torch_npu.testing.common_distributed import TEST_SKIPS, skipIfUnsupportMultiNPU

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

BACKEND = os.environ["BACKEND"]
TEMP_DIR = os.environ["TEMP_DIR"]
INIT_METHOD = os.getenv("INIT_METHOD", "env://")

DEFAULT_TIMEOUT = 300
CUSTOMIZED_TIMEOUT = {"test_DistributedDataParallel": 500}


class _FC2(nn.Module):
    def __init__(self):
        super(_FC2, self).__init__()
        self.fc = nn.Linear(10, 50, bias=True)
        self.fc.bias.requires_grad = False

    def forward(self, x):
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = _FC2()
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_param = nn.Parameter(torch.tensor([2, 2]).long(),
                                          requires_grad=False)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class BatchNormNet(nn.Module):

    def __init__(self):
        super(BatchNormNet, self).__init__()
        self.fc1 = nn.Linear(2, 40, bias=False)
        self.bn = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(40, 4, bias=False)

    def forward(self, x):
        x = torch.reshape(self.fc1(x), (-1, 4, 10))
        x = self.bn(x)
        x = torch.reshape(x, (-1, 40))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


DDP_NET = Net()
BN_NET = BatchNormNet()
ONLY_SBN_NET = nn.SyncBatchNorm(2, momentum=0.99)


def get_timeout(test_id):
    test_name = test_id.split(".")[-1]
    if test_name in CUSTOMIZED_TIMEOUT:
        return CUSTOMIZED_TIMEOUT[test_name]
    else:
        return DEFAULT_TIMEOUT


if not dist.is_available():
    print("Distributed not available, skipping tests")
    sys.exit(0)


@contextmanager
def _lock():
    lockfile = os.path.join(TEMP_DIR, "lockfile")
    with os.fdopen(os.open(lockfile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=0o600), "w") as lf:
        try:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            lf.close()


class Barrier(object):
    barrier_id = 0

    @classmethod
    def init(cls):
        cls.barrier_id = 0
        barrier_dir = os.path.join(TEMP_DIR, "barrier")
        for f_name in os.listdir(barrier_dir):
            os.unlink(os.path.join(barrier_dir, f_name))

    @classmethod
    def sync(cls, wait_for=None, timeout=30):
        if wait_for is None:
            wait_for = dist.get_world_size()
        cls.barrier_id += 1
        barrier_dir = os.path.join(TEMP_DIR, "barrier")
        pid = str(os.getpid())
        barrier_file = os.path.join(barrier_dir, pid)
        with _lock():
            with os.fdopen(os.open(barrier_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode=0o600), "w") as f:
                f.write(str(cls.barrier_id))

        start_time = time.time()
        while True:
            arrived = 0
            with _lock():
                for f_name in os.listdir(barrier_dir):
                    file_path = os.path.join(barrier_dir, f_name)
                    PathManager.check_directory_path_readable(file_path)
                    with open(file_path, "r") as f:
                        data = f.read()
                        if int(data) >= cls.barrier_id:
                            arrived += 1
            if arrived == wait_for:
                break

            if time.time() - start_time > timeout:
                raise RuntimeError("barrier timeout")
            time.sleep(0.1)

# [How does MultiProcessTestCase work?]
# Each MultiProcessTestCase instance uses 1 + `world_size()` processes, by
# default `world_size()` returns 4. Let's take `test_rpc_spawn.py` as an
# example which inherits from this class. Its `Setup()` methods calls into
# `MultiProcessTestCase._spawn_processes()` which spawns `world_size()`
# subprocesses. During the spawn, the main process passes the test name to
# subprocesses, and the name is acquired from self.id(). The subprocesses
# then use the provided test function name to retrieve the function attribute
# from the test instance and run it. The main process simply waits for all
# subprocesses to join.


class MultiProcessTestCase(TestCase):
    MAIN_PROCESS_RANK = -1
    # This exit code is used to indicate that the test code had an error and
    # exited abnormally. There are certain tests that might use sys.exit() to
    # simulate failures and in those cases, we can't have an exit code of 0,
    # but we still want to ensure we didn't run into any other errors.
    TEST_ERROR_EXIT_CODE = 10

    @property
    def world_size(self):
        return 4

    def join_or_run(self, fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                try:
                    fn()
                except Exception as e:
                    logging.error('Caught exception: \n%sexiting process with exit code: %s',
                                  traceback.format_exc(), MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
                    sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        return types.MethodType(wrapper, self)

    # The main process spawns N subprocesses that run the test.
    # Constructor patches current instance test method to
    # assume the role of the main process and join its subprocesses,
    # or run the underlying test function.
    def __init__(self, method_name='runTest'):
        super().__init__(method_name)
        fn = getattr(self, method_name)
        setattr(self, method_name, self.join_or_run(fn))

    def setUp(self):
        super().setUp()
        self.skip_return_code_checks = []
        self.processes = []
        self.rank = self.MAIN_PROCESS_RANK
        self.file_name = tempfile.NamedTemporaryFile(delete=False).name
        global TEST_SKIPS
        self.old_test_skips = TEST_SKIPS.copy()

    def tearDown(self):
        super().tearDown()
        for p in self.processes:
            p.terminate()
        # Each Process instance holds a few open file descriptors. The unittest
        # runner creates a new TestCase instance for each test method and keeps
        # it alive until the end of the entire suite. We must thus reset the
        # processes to prevent an effective file descriptor leak.
        self.processes = []

    def _current_test_name(self):
        # self.id() == e.g. '__main__.TestDistributed.TestAdditive.test_get_rank'
        return self.id().split(".")[-1]

    def _start_processes(self, proc):
        test_skips_manager = Manager()
        test_skips = test_skips_manager.dict()
        global TEST_SKIPS
        test_skips.update(TEST_SKIPS)
        TEST_SKIPS = test_skips

        self.processes = []
        for rank in range(int(self.world_size)):
            process = proc(
                target=self.__class__._run,
                name='process ' + str(rank),
                args=(rank, self._current_test_name(), self.file_name))
            process.start()
            self.processes.append(process)

    def _fork_processes(self):
        proc = torch.multiprocessing.get_context("fork").Process
        self._start_processes(proc)

    def _spawn_processes(self):
        proc = torch.multiprocessing.get_context("spawn").Process
        self._start_processes(proc)

    @classmethod
    def _run(cls, rank, test_name, file_name):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retrieving a corresponding test and executing it.
        getattr(self, test_name)()
        # exit to avoid run teardown() for fork processes
        sys.exit(0)

    def _join_processes(self, fn):
        timeout = get_timeout(self.id())
        start_time = time.time()
        subprocess_error = False
        try:
            while True:
                # check to see if any subprocess exited with an error early.
                for (i, p) in enumerate(self.processes):
                    # This is the exit code processes exit with if they
                    # encountered an exception.
                    if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE:
                        print("Process {} terminated with exit code {}, terminating remaining processes.".format(
                            i, p.exitcode))
                        active_children = torch.multiprocessing.active_children()
                        for ac in active_children:
                            ac.terminate()
                        subprocess_error = True
                        break
                if subprocess_error:
                    break
                # All processes have joined cleanly if they all a valid exitcode
                if all([p.exitcode is not None for p in self.processes]):
                    break
                # Check if we should time out the test. If so, we terminate each process.
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    print(
                        "Timing out after {} seconds and killing subprocesses.".format(
                            timeout
                        )
                    )
                    for p in self.processes:
                        p.terminate()
                    break
                # Sleep to avoid excessive busy polling.
                time.sleep(0.1)
            elapsed_time = time.time() - start_time
            if fn in self.skip_return_code_checks:
                self._check_no_test_errors(elapsed_time)
            else:
                self._check_return_codes(elapsed_time)
        finally:
            global TEST_SKIPS
            TEST_SKIPS = self.old_test_skips

    def _check_no_test_errors(self, elapsed_time):
        """
        Checks that we didn't have any errors thrown in the child processes.
        """
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError('Process {} timed out after {} seconds'.format(i, elapsed_time))
            self.assertNotEqual(self.TEST_ERROR_EXIT_CODE, p.exitcode)

    def _check_return_codes(self, elapsed_time):
        """
        Checks that the return codes of all spawned processes match, and skips
        tests if they returned a return code indicating a skipping condition.
        """
        first_process = self.processes[0]
        # first, we check if there are errors in actual processes
        # (via TEST_ERROR_EXIT CODE), and raise an exception for those.
        # the reason we do this is to attempt to raise a more helpful error
        # message than "Process x terminated/timed out"
        errored_processes = [
            (i, p)
            for i, p in enumerate(self.processes)
            if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE
        ]
        if errored_processes:
            error = "Processes {} exited with error code {}".format(
                " ".join([str(i) for (i, _) in errored_processes]),
                MultiProcessTestCase.TEST_ERROR_EXIT_CODE,
            )
            raise RuntimeError(error)
        # If no process exited uncleanly, we check for timeouts, and then ensure
        # each process exited cleanly.
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError('Process {} terminated or timed out after {} seconds'.format(i, elapsed_time))
            self.assertEqual(
                p.exitcode,
                first_process.exitcode
            )
        for skip in TEST_SKIPS.values():
            if first_process.exitcode == skip.exit_code:
                raise unittest.SkipTest(skip.message)
        self.assertEqual(
            first_process.exitcode,
            0
        )

    @property
    def is_master(self):
        return self.rank == 0


class _DistTestBase(object):
    def _barrier(self, *args, **kwargs):
        Barrier.sync(*args, **kwargs)

    def _init_group_test(self, **kwargs):
        group = [1, 2]
        group_id = dist.new_group(group, **kwargs)
        rank = dist.get_rank()
        if rank not in group:
            return ([], None, rank)

        return (group, group_id, rank)

    def _init_full_group_test(self, **kwargs):
        group = list(range(0, dist.get_world_size()))
        group_id = dist.new_group(**kwargs)
        rank = dist.get_rank()
        return (group, group_id, rank)

    def _init_global_test(self):
        group = list(range(0, dist.get_world_size()))
        group_id = dist.group.WORLD
        rank = dist.get_rank()
        return (group, group_id, rank)

    # HELPER FOR MULTINPU TESTS
    def _init_multinpu_helper(self):
        """Multinpu tests are designed to simulate the multi nodes with multi
        NPUs on each node. Hccl backend requires one NPU device in each process.
        """
        nNPUs = torch.npu.device_count()
        world_size = dist.get_world_size()
        visible_devices = range(min(nNPUs, world_size))

        nNPUs_per_process = 1
        rank_to_NPU = {
            i: list(
                visible_devices[i * nNPUs_per_process: (i + 1) * nNPUs_per_process]
            )
            for i in range(world_size)
        }
        return rank_to_NPU

    def _model_step(self, model):
        for param in model.parameters():
            if param.grad is not None:
                with torch.no_grad():
                    param += param.grad
                param.grad = None

    def _prepare_dummy_data(self, local_bs):
        # global_bs for DDP should be divisible by WORLD_SIZE
        global_bs = int(dist.get_world_size()) * local_bs
        input_cpu = torch.randn(global_bs, 2)
        target = torch.randn(global_bs, 4)
        loss = nn.MSELoss()
        return global_bs, input_cpu, target, loss

    # END TO END TEST FOR DISTRIBUTEDDATAPARALLEL
    def _test_DDP_helper(self, model, input_var, target, loss, scale_factor=1.0):
        model.train()
        output = model(input_var)
        out = loss(output, target) * scale_factor
        out.backward()

    def _assert_equal_param(self, param_npu, param_DDP):
        self.assertEqual(len(param_npu), len(param_DDP))
        for p_npu, p_DDP in zip(param_npu, param_DDP):
            self.assertEqual(p_npu, p_DDP)

    def _test_DDP_5iter(
            self, model_base, model_DDP, input_data, target, loss, local_bs, rank, batch_size, test_save,
            offset=None, world_size=0
    ):
        for idx in range(5):
            # single cpu/npu training
            self._test_DDP_helper(model_base, input_data, target, loss)

            if offset is None:
                offset = rank * local_bs

            # DDP training, DDP scatters subsets of input_cpu to nodes/NPUs
            self._test_DDP_helper(
                model_DDP,
                input_data[offset: offset + local_bs],
                target[offset: offset + local_bs],
                loss,
                world_size * local_bs / batch_size if world_size != 0 else 1,
            )

            # Update weights and run a second iteration to shake out errors
            self._model_step(model_base)
            self._model_step(model_DDP)
            self._assert_equal_param(
                list(model_base.parameters()), list(model_DDP.module.parameters())
            )

            # Shuffle the input so that DDP input is different
            input_data = input_data[torch.randperm(batch_size)]

            # save the model in the middle and reload
            if test_save and idx == 2 and INIT_METHOD.startswith("file://"):
                with tempfile.NamedTemporaryFile() as tmp:
                    state = {'net': model_DDP.state_dict()}
                    torch.save(state, tmp.name)
                    checkpoint = torch.load(tmp.name)
                    model_DDP.load_state_dict(checkpoint['net'])

        with tempfile.TemporaryFile() as tmp_file:
            state = {'net': model_DDP.state_dict()}
            torch.save(state, tmp_file)
            tmp_file.seek(0)
            checkpoint = torch.load(tmp_file)
            saved_model = copy.deepcopy(model_DDP)
            saved_model.load_state_dict(checkpoint['net'])
        for k in model_DDP.state_dict():
            self.assertEqual(model_DDP.state_dict()[k],
                             saved_model.state_dict()[k])

    def _test_DistributedDataParallel(self, npu_subset, rank, bucket_view, output_device=None):
        # Run a simple end to end DDP model, use result of single node model
        # as baseline

        # cpu training setup
        model = DDP_NET

        # single npu training setup
        model_npu = copy.deepcopy(model)
        model_npu.npu(npu_subset[0])

        # DDP training setup
        model_DDP = copy.deepcopy(model)
        model_DDP.npu(npu_subset[0])

        model_DDP = nn.parallel.DistributedDataParallel(
            model_DDP, device_ids=npu_subset, gradient_as_bucket_view=bucket_view
        )

        # test serializable/unserializable
        with tempfile.NamedTemporaryFile() as tmp:
            state = {'net': model_DDP.state_dict()}
            torch.save(state, tmp.name)
            checkpoint = torch.load(tmp.name)
            model_DDP.load_state_dict(checkpoint['net'])

        # dummy data initialization
        local_bs = len(npu_subset)
        global_bs, input_cpu, target, loss = self._prepare_dummy_data(local_bs)

        # check two model parameters over 5 iterations
        self._test_DDP_5iter(
            model_npu,
            model_DDP,
            input_cpu.npu(npu_subset[0]),
            target.npu(npu_subset[0]),
            loss,
            local_bs,
            rank,
            global_bs,
            True
        )
        self._barrier()

    def test_DistributedDataParallel_requires_grad(self):
        # a module without gradients shouldn't be accepted
        self.assertRaises(RuntimeError, lambda: nn.parallel.DistributedDataParallel(nn.Module()))

    def test_DistributedDataParallel(self):
        group, group_id, rank = self._init_global_test()
        rank_to_NPU = self._init_multinpu_helper()
        npus = list(rank_to_NPU[rank])
        for bk in [True, False]:
            self._test_DistributedDataParallel(npu_subset=npus, rank=rank, bucket_view=bk)


    def _test_DistributedDataParallel_SyncBatchNorm(
            self, npu_subset, rank, local_bs, global_bs, offset, bucket_view, output_device=None):
        # Run a simple end to end DDP model, use result of single node model
        # as baseline

        # cpu training setup
        model = BN_NET

        # single npu training setup
        model_npu = copy.deepcopy(model)
        model_npu.npu(npu_subset[0])

        # DDP training setup
        model_DDP = nn.SyncBatchNorm.convert_sync_batchnorm(copy.deepcopy(model))
        model_DDP.npu(npu_subset[0])
        model_DDP = nn.parallel.DistributedDataParallel(
            model_DDP, device_ids=npu_subset, gradient_as_bucket_view=bucket_view
        )

        # test serializable/unserializable
        with tempfile.NamedTemporaryFile() as tmp:
            state = {'net': model_DDP.state_dict()}
            torch.save(state, tmp.name)
            checkpoint = torch.load(tmp.name)
            model_DDP.load_state_dict(checkpoint['net'])

        # data initialization
        input_cpu = torch.randn(global_bs, 2)
        target = torch.randn(global_bs, 4)
        loss = nn.MSELoss()

        # check two model parameters over 5 iterations
        self._test_DDP_5iter(
            model_npu,
            model_DDP,
            input_cpu.npu(npu_subset[0]),
            target.npu(npu_subset[0]),
            loss,
            local_bs,
            rank,
            global_bs,
            True,
            offset,
            int(WORLD_SIZE)
        )
        self._barrier()

    def test_DistributedDataParallel_SyncBatchNorm(self):
        group, group_id, rank = self._init_global_test()
        # DDP does not support replicating BN layers within a process, hence
        # testing with one module replica per process
        npus = [rank]

        num_processes = int(WORLD_SIZE)
        local_bs = 2
        bs_offset = int(rank * 2)
        global_bs = int(num_processes * 2)
        for bk in [True, False]:
            self._test_DistributedDataParallel_SyncBatchNorm(
                npu_subset=npus,
                rank=rank,
                local_bs=local_bs,
                global_bs=global_bs,
                offset=bs_offset,
                bucket_view=bk)

    def _test_DistributedDataParallel_SyncBatchNorm_2D_Input(self, bucket_view):
        group, group_id, rank = self._init_global_test()
        # DDP does not support replicating BN layers within a process, hence
        # testing with one module replica per process
        npus = [rank]

        model = nn.BatchNorm1d(2)

        # single npu training setup
        model_npu = copy.deepcopy(model)
        model_npu.npu(npus[0])

        # DDP training setup
        model_DDP = nn.SyncBatchNorm.convert_sync_batchnorm(copy.deepcopy(model))
        model_DDP.npu(npus[0])
        model_DDP = nn.parallel.DistributedDataParallel(
            model_DDP, device_ids=npus, gradient_as_bucket_view=bucket_view
        )

        local_bs = len(npus) * 2
        global_bs = int(WORLD_SIZE) * local_bs
        input_cpu = torch.randn(global_bs, 2)
        target = torch.randn(global_bs, 2)
        loss = nn.MSELoss()

        # check two model parameters over 5 iterations
        self._test_DDP_5iter(
            model_npu,
            model_DDP,
            input_cpu.npu(npus[0]),
            target.npu(npus[0]),
            loss,
            local_bs,
            rank,
            global_bs,
            True
        )
        self._barrier()

    def test_DistributedDataParallel_SyncBatchNorm_2D_Input(self):
        for bk in [True, False]:
            self._test_DistributedDataParallel_SyncBatchNorm_2D_Input(bk)

    def _test_DistributedDataParallel_SyncBatchNorm_Diff_Input_Sizes_Running_Value(self, bucket_view):
        group, group_id, rank = self._init_global_test()
        model = nn.parallel.DistributedDataParallel(ONLY_SBN_NET.npu(rank), device_ids=[rank],
                                                    gradient_as_bucket_view=bucket_view)

        input_var = []
        for i in range(int(WORLD_SIZE)):
            input_var_rank = torch.cat([
                torch.ones(2, 1, 10 ** (i + 1)) * (0.1 ** (i - 1)),
                torch.ones(2, 1, 10 ** (i + 1)) * (0.3 ** (i - 1))
            ], dim=1)
            input_var.append(input_var_rank)

        all_input_var = torch.cat(
            [x.permute(1, 0, 2).contiguous().view(ONLY_SBN_NET.num_features, -1) for x in input_var],
            dim=1
        ).npu(rank)

        for i in range(5):
            y = model(input_var[rank].npu(rank))
            y.mean().backward()

        running_mean, running_var = model.module.running_mean, model.module.running_var
        torch.testing.assert_allclose(running_mean, all_input_var.mean(1))
        torch.testing.assert_allclose(running_var.cpu(), all_input_var.cpu().var(1, unbiased=False))

    # need more 4 device, less 4 divice there may be accuracy issues 
    @skipIfUnsupportMultiNPU(4)
    def test_DistributedDataParallel_SyncBatchNorm_Diff_Input_Sizes_Running_Value(self):
        for bk in [True, False]:
            self._test_DistributedDataParallel_SyncBatchNorm_Diff_Input_Sizes_Running_Value(bk)

    def test_DistributedDataParallel_SyncBatchNorm_Diff_Input_Sizes_gradient(self):
        group, group_id, rank = self._init_global_test()
        # only do single NPU per process
        npus = [rank]

        num_processes = int(WORLD_SIZE)
        local_bs = rank + 2
        bs_offset = int((rank + 3) * rank / 2)
        global_bs = int((num_processes + 3) * num_processes / 2)
        for bk in [True, False]:
            self._test_DistributedDataParallel_SyncBatchNorm(
                npu_subset=npus,
                rank=rank,
                local_bs=local_bs,
                global_bs=global_bs,
                offset=bs_offset,
                bucket_view=bk)

    @skipIfNoTorchVision
    def test_SyncBatchNorm_process_group(self):
        # When adopting `convert_sync_batchnorm` to convert a `nn.modules`,
        # it need to recursively pass the `process_group` in the module when the `SyncBatchNorm`
        # is nested in a sub-module or sub-sub-module (e.g. resnet50 in torchvision.models).

        process_ids = 0
        process_group = torch.distributed.new_group([process_ids])
        res50_model = torchvision.models.resnet50()
        res50_model_sync = nn.SyncBatchNorm.convert_sync_batchnorm(copy.deepcopy(res50_model), process_group)
        process_group_sync = res50_model_sync.layer1[0].bn1.process_group
        self.assertEqual(process_group_sync, process_group)


FILE_SCHEMA = "file://"
tmp_dir = None


def initialize_temp_directories(init_method=None):
    global tmp_dir
    tmp_dir = tempfile.TemporaryDirectory()
    os.environ["TEMP_DIR"] = tmp_dir.name
    os.mkdir(os.path.join(tmp_dir.name, "barrier"))
    os.mkdir(os.path.join(tmp_dir.name, "test_dir"))
    init_dir_path = os.path.join(tmp_dir.name, "init_dir")
    os.mkdir(init_dir_path)


def cleanup_temp_dir():
    if tmp_dir is not None:
        tmp_dir.cleanup()


WORLD_SIZE = os.environ["WORLD_SIZE"]


class TestDistBackend(MultiProcessTestCase, _DistTestBase):
    @classmethod
    def setUpClass(cls):
        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(MASTER_PORT)
        super().setUpClass()

    def setUp(self):
        super().setUp()
        # initialize temp directories
        initialize_temp_directories()
        # initialize Barrier
        Barrier.init()
        self._spawn_processes()

    def tearDown(self):
        cleanup_temp_dir()
        super().tearDown()

    @property
    def init_method(self):
        return "{}{file_name}".format(FILE_SCHEMA, file_name=self.file_name)

    @classmethod
    def _run(cls, rank, test_name, file_name):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        torch.npu.set_device(rank)

        if torch.npu.is_available() and torch.npu.device_count() < int(self.world_size):
            sys.exit(TEST_SKIPS['multi-npu'].exit_code)
        try:
            timeout = timedelta(seconds=60)
            dist.init_process_group(
                init_method=INIT_METHOD,
                backend=BACKEND,
                world_size=int(self.world_size),
                rank=self.rank,
                timeout=timeout,
            )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        self._barrier()

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retreiving a corresponding test and executing it.
        getattr(self, test_name)()
        self._barrier()
        dist.destroy_process_group()
        sys.exit(0)

    # Needed since MultiProcessTestCase assumes a world_size of 4, but we
    # run these tests under other various world_sizes.
    @property
    def world_size(self):
        return os.environ["WORLD_SIZE"]


if __name__ == "__main__":
    run_tests()
