from typing import Any, Callable, Dict, Optional, Sequence
from unittest import skip

import torch

import torch.utils._pytree as pytree
from torch import Tensor

from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed._tensor.placement_types import (
    _Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.distributed_c10d import ReduceOp
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase

import torch_npu
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU


def no_op():
    return None


def deepcopy_convert_to_dtensor(
    val: Any,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> Any:
    """
    Recursively convert (over Sequence and Dict types) Tensors into DTensors.

    :param device_mesh: the DeviceMesh to use.
    :param placements: the Placement list to use.
    :return: the transformed structure.
    """

    def f(x):
        if isinstance(x, Tensor) and not isinstance(x, DTensor):
            return distribute_tensor(
                x,
                device_mesh=device_mesh,
                placements=placements,
            )
        return x

    return pytree.tree_map(f, [val])[0]


def deepcopy_convert_from_dtensor(val: Any) -> Any:
    """
    Recursive convert any DTensor to local Tensor.

    :param val: the structure to coerce.
    :return: the coerced structure.
    """

    def f(x):
        if isinstance(x, DTensor):
            return x.redistribute(
                device_mesh=x.device_mesh,
                placements=[Replicate()] * x.device_mesh.ndim,
            ).to_local()
        return x

    return pytree.tree_map(f, [val])[0]


class DistElementwiseOpsTest(DTensorTestBase):
    def _compare_pairwise_ops(
        self,
        *,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        op: Callable,
        pre_op_fn: Optional[Callable] = None,
        args: Sequence[Any] = tuple(),
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        if pre_op_fn is None:
            pre_op_fn = no_op

        if not kwargs:
            kwargs = {}

        dargs = deepcopy_convert_to_dtensor(
            args,
            device_mesh=device_mesh,
            placements=placements,
        )
        dkwargs = deepcopy_convert_to_dtensor(
            kwargs,
            device_mesh=device_mesh,
            placements=placements,
        )
        pre_op_fn()

        # run the reference first, in case the call is broken;
        # it's better to debug an incorrect call at this point.
        reference_result = op(*args, **kwargs)
        pre_op_fn()
        dist_result = op(*dargs, **dkwargs)
        collected_result = deepcopy_convert_from_dtensor(dist_result)
        self.assertEqual(reference_result, collected_result)

    def _run_sharded_elementwise_ops(
        self,
        *,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        pre_op_fn: Optional[Callable] = None,
        input_size: Sequence[int],
        op: Callable,
        **kwargs,
    ):
        if pre_op_fn is None:
            pre_op_fn = no_op

        input_tensor = torch.randn(
            *input_size,
            device=self.device_type,
            requires_grad=True,
        )
        self._compare_pairwise_ops(
            device_mesh=device_mesh,
            placements=placements,
            pre_op_fn=pre_op_fn,
            op=op,
            args=(input_tensor,),
            kwargs=kwargs,
        )

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_partial_add(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        d_1 = DTensor.from_local(torch.rand(2, 2), device_mesh, [_Partial()])
        d_2 = DTensor.from_local(torch.rand(2, 2), device_mesh, [_Partial()])
        d_3 = d_1 + d_2
        self.assertEqual(d_3._spec.placements[0].is_partial(), True)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_activations(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(0)],
            input_size=(8, 5),
            op=torch.nn.functional.gelu,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Replicate()],
            input_size=(8, 5),
            op=torch.nn.functional.gelu,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(1)],
            input_size=(3, 12),
            op=torch.nn.functional.relu,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Replicate()],
            input_size=(8, 5),
            op=torch.nn.functional.relu,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(0)],
            input_size=(8, 5),
            op=torch.sigmoid,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Replicate()],
            input_size=(8, 5),
            op=torch.sigmoid,
        )

    @skipIfUnsupportMultiNPU(4)
    @skip("testing RNG based ops is broken")
    @with_comms
    def test_dropout(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        def _reset_random_seed():
            torch.manual_seed(self.rank + 4)

        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(0)],
            input_size=(8, 5),
            op=torch.nn.functional.dropout,
            pre_op_fn=_reset_random_seed,
            p=0.4,
            training=False,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(1)],
            input_size=(3, 14),
            op=torch.nn.functional.dropout,
            pre_op_fn=_reset_random_seed,
            p=0.5,
            training=True,
        )

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_dropout_errors(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        with self.assertRaisesRegex(RuntimeError, "supported"):
            self._run_sharded_elementwise_ops(
                device_mesh=device_mesh,
                placements=[_Partial(ReduceOp.SUM)],
                input_size=(8, 5),
                op=torch.nn.functional.dropout,
            )

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_mul_out(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        torch.manual_seed(self.rank)
        shard_spec = [Shard(0)]
        input_size = (8, 4)
        input_tensor = torch.randn(*input_size, device=self.device_type)
        dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)

        other_tensor = torch.randn(*input_size, device=self.device_type)
        other_dtensor = DTensor.from_local(other_tensor, device_mesh, shard_spec)

        output_tensor = torch.randn(*input_size, device=self.device_type)
        output_dtensor = DTensor.from_local(output_tensor, device_mesh, shard_spec)
        dt = torch.mul(dtensor, other_dtensor, out=output_dtensor)
        expected = torch.mul(input_tensor, other_tensor, out=output_tensor)
        self.assertEqual(input_tensor, dtensor.to_local())
        self.assertEqual(expected, dt.to_local())


if __name__ == "__main__":
    run_tests()
