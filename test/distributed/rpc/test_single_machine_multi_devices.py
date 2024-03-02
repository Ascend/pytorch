import os
import time
import unittest
import warnings
import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import multiprocessing as mp
from torch import nn, Tensor
from torch.distributed.nn.api.remote_module import RemoteModule
from torch.distributed.rpc import WorkerInfo, PyRRef

import torch_npu
from torch_npu.distributed.rpc.options import NPUTensorPipeRpcBackendOptions
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests


class TestRpc(TestCase):
    world_size_2p = 2
    world_size_3p = 3

    def setUp(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29501'

    @classmethod
    def echo(cls, input1, input2):
        return (input1, input2)

    @classmethod
    @rpc.functions.async_execution
    def async_add_chained(cls, to, x, y, z):
        # This function runs on "worker1" and returns immediately when
        # the callback is installed through the `then(cb)` API. In the
        # mean time, the `rpc_async` to "worker2" can run concurrently.
        # When the return value of that `rpc_async` arrives at
        # "worker1", "worker1" will run the lambda function accordingly
        # and set the value for the previously returned `Future`, which
        # will then trigger RPC to send the result back to "worker0".
        return rpc.rpc_async(to, TestRpc.echo, args=(x, y)).then(
            lambda fut: fut.wait()
        )

    @classmethod
    def set_options(cls):
        options = NPUTensorPipeRpcBackendOptions(num_worker_threads=8, device_maps={'worker1': {'npu:0': 'npu:1'}})
        options.rpc_timeout = 60
        options.num_worker_threads = 16
        options.set_devices(['npu:0'])
        return options

    @classmethod
    def init_worker_info(cls, process_id):
        npu_id_ = f'npu:{process_id}'
        worker_name_ = f'worker{process_id}'
        torch.npu.set_device(npu_id_)
        warnings.filterwarnings('ignore', category=UserWarning)
        return npu_id_, worker_name_

    @classmethod
    def _test_async_call_for_cpu(cls, pid, inputs, world_size):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)

            times = []
            for i in range(2):
                start = time.time()
                fut = rpc.rpc_async('worker1', TestRpc.echo, args=(inputs[i], inputs[i]))
                rets = fut.wait()
                times.append(time.time() - start)
            print('test_async_call_for_cpu_1M cost:', times[0], 's test_async_call_for_cpu_1G cost:', times[1], 's')
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_sync_call_for_cpu(cls, pid, inputs, world_size):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)

            times = []
            for i in range(2):
                start = time.time()
                rets = rpc.rpc_sync('worker1', TestRpc.echo, args=(inputs[i], inputs[i]))
                times.append(time.time() - start)
            print('test_sync_call_for_cpu_1M cost:', times[0], 's test_sync_call_for_cpu_1G cost:', times[1], 's')
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_async_call_for_npu(cls, pid, inputs, world_size):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)
            times = []
            for i in range(2):
                input1 = inputs[i].npu()
                start = time.time()
                fut = rpc.rpc_async('worker1', TestRpc.echo, args=(input1, input1))
                rets = fut.wait()
                times.append(time.time() - start)
            print('test_async_call_for_npu_1M cost:', times[0], 's test_async_call_for_npu_1G cost:', times[1], 's')
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_sync_call_for_npu(cls, pid, inputs, world_size):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)

            times = []
            for i in range(2):
                input1 = inputs[i].npu()
                start = time.time()
                rets = rpc.rpc_sync('worker1', TestRpc.echo, args=(input1, input1))
                times.append(time.time() - start)
            print('test_sync_call_for_npu_1M cost:', times[0], 's test_sync_call_for_npu_1G cost:', times[1], 's')
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_remote_rref(cls, pid, inputs, world_size):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            options.set_device_map('worker2', {'npu:0': 'npu:2'})
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)

            times = []
            for i in range(2):
                input1 = inputs[i].npu()
                start = time.time()
                rref1 = rpc.remote('worker1', TestRpc.echo, args=(input1, inputs[2]))
                rref2 = rpc.remote('worker2', TestRpc.echo, args=(input1, inputs[3]))
                rref1.to_here()
                rref2.to_here()
                times.append(time.time() - start)
            print('test_remote_rref_1M cost:', times[0], 's test_remote_rref_1G cost:', times[1], 's')
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_async_execution(cls, pid, inputs, world_size):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            options.set_device_map('worker2', {'npu:0': 'npu:2'})
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)

            times = []
            for i in range(2):
                input1 = inputs[i].npu()
                start = time.time()
                ret = rpc.rpc_sync('worker1', TestRpc.async_add_chained,
                                   args=('worker2', input1, inputs[2], inputs[3]))
                times.append(time.time() - start)
            print('test_async_execution_1M cost:', times[0], 's test_async_execution_1G cost:', times[1], 's')
        elif pid == 1:
            options = NPUTensorPipeRpcBackendOptions(num_worker_threads=8, device_maps={'worker2': {'npu:1': 'npu:2'}})
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_get_worker_info(cls, pid, inputs, world_size):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)

            start = time.time()
            worker_info = rpc.get_worker_info(worker_name_)
            print('test_get_worker_info cost:', time.time() - start, 's')
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_dist_autograd_sync_cpu(cls, pid, inputs, world_size):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)

            with dist_autograd.context() as context_id:
                t1 = torch.rand((3, 3), requires_grad=True)
                t2 = torch.rand((3, 3), requires_grad=True)
                t3 = rpc.rpc_sync('worker1', torch.add, args=(t1, t2))
                t4 = torch.rand((3, 3), requires_grad=True)
                t5 = torch.mul(t3, t4)
                loss = t5.sum()
                dist_autograd.backward(context_id, [loss])
                dist_autograd.get_gradients(context_id)
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_dist_autograd_sync_npu(cls, pid, inputs, world_size):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)

            with dist_autograd.context() as context_id:
                t1 = torch.rand((3, 3), requires_grad=True).npu()
                t2 = torch.rand((3, 3), requires_grad=True).npu()
                t3 = rpc.rpc_sync('worker1', torch.add, args=(t1, t2))
                t4 = torch.rand((3, 3), requires_grad=True).npu()
                t5 = torch.mul(t3, t4)
                loss = t5.sum()
                dist_autograd.backward(context_id, [loss])
                dist_autograd.get_gradients(context_id)
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_remote_pyrref_api(cls, pid, inputs, world_size):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)
            t1 = torch.ones(3, requires_grad=True).npu()
            t2 = torch.ones(3, requires_grad=True).npu()
            pyrref = rpc.remote('worker1', torch.add, args=(t1, t2))
            TestCase().assertEqual(pyrref.rpc_async().size().wait(), torch.Size([3]))
            TestCase().assertEqual(pyrref.rpc_sync().size(), torch.Size([3]))
            TestCase().assertEqual(pyrref.to_here().sum(), torch.tensor(6))
            TestCase().assertEqual(pyrref.owner_name(), 'worker1')
            TestCase().assertEqual(pyrref.is_owner(), False)
            TestCase().assertEqual(pyrref.confirmed_by_owner(), True)
            TestCase().assertEqual(pyrref.remote().size().to_here(), torch.Size([3]))
            TestCase().assertEqual(pyrref.owner(), WorkerInfo(id=1, name='worker1'))
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_get_module_rref(cls, pid, inputs, world_size):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)

            remote_linear_module = RemoteModule("worker1/npu:1", nn.Linear, args=(20, 30), )
            TestCase().assertEqual(remote_linear_module.get_module_rref(), remote_linear_module.module_rref)
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_remote_parameters(cls, pid, inputs, world_size):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)

            remote_linear_module = RemoteModule("worker1/npu:1", nn.Linear, args=(20, 30), )
            param_rrefs = remote_linear_module.remote_parameters()
            TestCase().assertEqual(len(param_rrefs), 2)
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_local_value(cls, pid, inputs, world_size):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)

            t1 = torch.ones(10, requires_grad=True).npu()
            pyrref = rpc.PyRRef(t1.sum() + t1.sum())
            pyrref.backward()
            TestCase().assertEqual(pyrref.local_value(), torch.tensor(20))
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    def _test_multiprocess(self, f, inputs, world_size):
        ctx = mp.get_context('spawn')
        ps = []
        for i in range(world_size):
            p = ctx.Process(target=f, args=(i, inputs, world_size))
            p.start()
            ps.append(p)

        for p in ps:
            p.join()

    @skipIfUnsupportMultiNPU(2)
    def test_async_call_for_cpu(self):
        inputs = [torch.rand(1024, 1024).cpu(), torch.rand(1024, 1024, 1024).cpu()]
        self._test_multiprocess(TestRpc._test_async_call_for_cpu, inputs, self.world_size_2p)

    @skipIfUnsupportMultiNPU(2)
    def test_sync_call_for_cpu(self):
        inputs = [torch.rand(1024, 1024).cpu(), torch.rand(1024, 1024, 1024).cpu()]
        self._test_multiprocess(TestRpc._test_sync_call_for_cpu, inputs, self.world_size_2p)

    @skipIfUnsupportMultiNPU(2)
    def test_async_call_for_npu(self):
        inputs = [torch.rand(1024, 1024), torch.rand(1024, 1024, 1024)]
        self._test_multiprocess(TestRpc._test_async_call_for_npu, inputs, self.world_size_2p)

    @skipIfUnsupportMultiNPU(2)
    def test_sync_call_for_npu(self):
        inputs = [torch.rand(1024, 1024), torch.rand(1024, 1024, 1024)]
        self._test_multiprocess(TestRpc._test_sync_call_for_npu, inputs, self.world_size_2p)

    @skipIfUnsupportMultiNPU(3)
    def test_remote_rref(self):
        inputs = [torch.rand(1024, 1024), torch.rand(1024, 1024, 1024), 3, 1]
        self._test_multiprocess(TestRpc._test_remote_rref, inputs, self.world_size_3p)

    @skipIfUnsupportMultiNPU(3)
    def test_async_execution(self):
        inputs = [torch.rand(1024, 1024), torch.rand(1024, 1024, 1024), 1, 1]
        self._test_multiprocess(TestRpc._test_async_execution, inputs, self.world_size_3p)

    @skipIfUnsupportMultiNPU(3)
    def test_get_worker_info(self):
        self._test_multiprocess(TestRpc._test_get_worker_info, [], self.world_size_3p)

    @skipIfUnsupportMultiNPU(2)
    def test_dist_autograd_sync_cpu(self):
        self._test_multiprocess(TestRpc._test_dist_autograd_sync_cpu, [], self.world_size_2p)

    @unittest.skip("NPU doesn't support yet.")
    @skipIfUnsupportMultiNPU(2)
    def test_dist_autograd_sync_npu(self):
        self._test_multiprocess(TestRpc._test_dist_autograd_sync_npu, [], self.world_size_2p)

    @skipIfUnsupportMultiNPU(2)
    def test_remote_pyrref_npu(self):
        self._test_multiprocess(TestRpc._test_remote_pyrref_api, [], self.world_size_2p)

    @skipIfUnsupportMultiNPU(2)
    def test_get_module_rref_npu(self):
        self._test_multiprocess(TestRpc._test_get_module_rref, [], self.world_size_2p)

    @skipIfUnsupportMultiNPU(2)
    def test_remote_parameters_npu(self):
        self._test_multiprocess(TestRpc._test_remote_parameters, [], self.world_size_2p)

    @skipIfUnsupportMultiNPU(2)
    def test_local_value_npu(self):
        self._test_multiprocess(TestRpc._test_local_value, [], self.world_size_2p)


if __name__ == '__main__':
    run_tests()
