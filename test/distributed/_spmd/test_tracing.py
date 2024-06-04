# Owner(s): ["oncall: distributed"]

from functools import wraps

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed._tensor import DeviceMesh
from torch.distributed.distributed_c10d import get_global_rank, get_world_size
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase
import torch_npu
from torch_npu.testing.common_distributed import with_comms as base_with_comms


def with_comms(func):
    @base_with_comms
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # make sure we set different random seeds for each rank
        # otherwise we dont need DDP / SPMD
        # (we would have the same parameters and inputs everywhere)
        torch.manual_seed(self.rank)
        return func(self, *args, **kwargs)

    return wrapper


class TraceDeviceMeshTestBase:
    def _test_tracing_all_reduce_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # check all dim groups
        dim_to_subgroups = mesh.get_group()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            def fn(tensor: torch.Tensor):
                tensor = funcol.all_reduce(tensor, "sum", group=(mesh, dim))
                # multiply with 1 to trigger wait on read during tracing.
                return tensor * 1

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)

            # execute traced DeviceMesh communication
            reduced_tensor = traced_fn(local_tensor.clone())
            res_num = sum(global_ranks)
            self.assertEqual(reduced_tensor, torch.ones(3, 3) * res_num)


class TraceDeviceMesh3DTest(DTensorTestBase, TraceDeviceMeshTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_tracing_all_reduce_nd(self):
        self._test_tracing_all_reduce_nd(torch.arange(8).reshape(2, 2, 2))


class TraceDeviceMesh2DTest(DTensorTestBase, TraceDeviceMeshTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_tracing_all_reduce_nd(self):
        self._test_tracing_all_reduce_nd(torch.arange(4).reshape(2, 2))


if __name__ == "__main__":
    run_tests()
