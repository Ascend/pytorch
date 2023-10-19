import os
import time
import warnings
import torch
import torch.distributed as dist
from torch import multiprocessing as mp
import torch.distributed.rpc as rpc
import torch_npu
from torch_npu.distributed.rpc.options import NPUTensorPipeRpcBackendOptions
from torch_npu.testing.common_utils import skipIfUnsupportMultiNPU
from torch_npu.testing.testcase import TestCase, run_tests


class TestRpc(TestCase):
    world_size_2p = 2
    world_size_3p = 3

    def setUp(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29501'

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
        return rpc.rpc_async(to, torch.add, args=(x, y)).then(
            lambda fut: fut.wait() + z
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
    def _test_async_call_for_cpu(cls, pid, inputs, world_size, c2p):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)

            times = []
            for i in range(2):
                start = time.time()
                fut = rpc.rpc_async('worker1', torch.add, args=(inputs[i], inputs[i]))
                rets = fut.wait()
                times.append(time.time() - start)
                c2p.put((rets, torch.add(inputs[i], inputs[i])))
            print('test_async_call_for_cpu_1M cost:', times[0], 's test_async_call_for_cpu_1G cost:', times[1], 's')
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_sync_call_for_cpu(cls, pid, inputs, world_size, c2p):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)

            times = []
            for i in range(2):
                start = time.time()
                rets = rpc.rpc_sync('worker1', torch.add, args=(inputs[i], inputs[i]))
                times.append(time.time() - start)
                c2p.put((rets, torch.add(inputs[i], inputs[i])))
            print('test_sync_call_for_cpu_1M cost:', times[0], 's test_sync_call_for_cpu_1G cost:', times[1], 's')
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_async_call_for_npu(cls, pid, inputs, world_size, c2p):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)
            times = []
            for i in range(2):
                start = time.time()
                fut = rpc.rpc_async('worker1', torch.add, args=(inputs[i].npu(), inputs[i].npu()))
                rets = fut.wait()
                times.append(time.time() - start)
                c2p.put((rets.cpu(), torch.add(inputs[i], inputs[i])))
            print('test_async_call_for_npu_1M cost:', times[0], 's test_async_call_for_npu_1G cost:', times[1], 's')
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_sync_call_for_npu(cls, pid, inputs, world_size, c2p):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)

            times = []
            for i in range(2):
                start = time.time()
                rets = rpc.rpc_sync('worker1', torch.add, args=(inputs[i].npu(), inputs[i].npu()))
                times.append(time.time() - start)
                c2p.put((rets.cpu(), torch.add(inputs[i], inputs[i])))
            print('test_sync_call_for_npu_1M cost:', times[0], 's test_sync_call_for_npu_1G cost:', times[1], 's')
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_remote_rref(cls, pid, inputs, world_size, c2p):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            options.set_device_map('worker2', {'npu:0': 'npu:2'})
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)

            times = []
            for i in range(2):
                start = time.time()
                rref1 = rpc.remote('worker1', torch.add, args=(inputs[i].npu(), inputs[2]))
                rref2 = rpc.remote('worker2', torch.add, args=(inputs[i].npu(), inputs[3]))
                rref = rref1.to_here() + rref2.to_here()
                times.append(time.time() - start)
                c2p.put((rref.cpu(), torch.add(torch.add(inputs[i], inputs[2]), torch.add(inputs[i], inputs[3]))))
            print('test_remote_rref_1M cost:', times[0], 's test_remote_rref_1G cost:', times[1], 's')
        else:
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE)
        rpc.shutdown()

    @classmethod
    def _test_async_execution(cls, pid, inputs, world_size, c2p):
        npu_id_, worker_name_ = TestRpc.init_worker_info(pid)
        if pid == 0:
            options = TestRpc.set_options()
            options.set_device_map('worker2', {'npu:0': 'npu:2'})
            rpc.init_rpc(worker_name_, rank=pid, world_size=world_size,
                         backend=rpc.backend_registry.BackendType.NPU_TENSORPIPE, rpc_backend_options=options)

            times = []
            for i in range(2):
                start = time.time()
                ret = rpc.rpc_sync('worker1', TestRpc.async_add_chained,
                                   args=('worker2', inputs[i].npu(), inputs[2], inputs[3]))
                times.append(time.time() - start)
                c2p.put((ret.cpu(), torch.add(inputs[i], torch.add(inputs[2], inputs[3]))))
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
    def _test_get_worker_info(cls, pid, inputs, world_size, c2p):
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

    def _test_multiprocess(self, f, inputs, world_size):
        ctx = mp.get_context('spawn')
        ps = []
        c2p = ctx.Queue(1)
        for i in range(world_size):
            p = ctx.Process(target=f, args=(i, inputs, world_size, c2p))
            p.start()
            ps.append(p)

        if inputs:
            result_1M, expect_1M = c2p.get()
            result_1G, expect_1G = c2p.get()
            msg1 = f'Expect 1M result:{expect_1M},but actual is {result_1M}'
            msg2 = f'Expect 1G result:{expect_1G},but actual is {result_1G}'
            self.assertEqual(result_1M, expect_1M, message=msg1)
            self.assertEqual(result_1G, expect_1G, message=msg2)

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


if __name__ == '__main__':
    run_tests()
