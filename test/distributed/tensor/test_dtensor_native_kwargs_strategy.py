# Regression tests: DTensor kwargs strategy path should work with native PyTorch 2.11+ support.
import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    TupleStrategy,
)
from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Replicate
from torch.testing._internal.common_utils import TestCase, run_tests


def _cpu_mesh(size: int = 4) -> DeviceMesh:
    return DeviceMesh("cpu", torch.arange(size))


def _replicate_strategy(mesh: DeviceMesh, shape: tuple[int, ...]) -> OpStrategy:
    placements = tuple(Replicate() for _ in range(mesh.ndim))
    strides: list[int] = []
    st = 1
    for s in reversed(shape):
        strides.append(st)
        st *= s
    strides_t = tuple(reversed(strides))
    meta = TensorMeta(shape=torch.Size(shape), stride=strides_t, dtype=torch.float32)
    spec = DTensorSpec(mesh=mesh, placements=placements, tensor_meta=meta)
    op_spec = OpSpec(output_specs=spec, input_specs=(spec,), redistribute_cost=[[0.0]])
    return OpStrategy([op_spec])


class TestNativeKwargsStrategyAndExpand(TestCase):
    """Mimics torch_npu _matrix_ops / _math_ops usage: args_strategy + kwargs_strategy length drives expand."""

    def setUp(self):
        from torch.testing._internal.distributed.fake_pg import FakeStore

        super().setUp()
        self.ws = 4
        torch.distributed.init_process_group(
            backend="fake", rank=0, world_size=self.ws, store=FakeStore()
        )

    def tearDown(self):
        torch.distributed.destroy_process_group()
        super().tearDown()

    def test_op_schema_kwargs_strategy_flat(self):
        mesh = _cpu_mesh(self.ws)
        a = _replicate_strategy(mesh, (2, 3))
        b = _replicate_strategy(mesh, (2, 3))
        schema = OpSchema(
            torch.ops.aten.add.Tensor,
            (a,),
            {"other": b, "dim": 0},
            RuntimeSchemaInfo(needs_pytree=False),
        )
        ks = schema.kwargs_strategy
        self.assertEqual(len(ks), 1)
        self.assertIs(ks[0], b)

    def test_op_schema_kwargs_strategy_pytree_tuple_in_kwarg(self):
        mesh = _cpu_mesh(self.ws)
        c0 = _replicate_strategy(mesh, (1, 1))
        c1 = _replicate_strategy(mesh, (2, 2))
        ts = TupleStrategy([c0, c1])
        schema = OpSchema(
            torch.ops.aten.add.Tensor,
            (_replicate_strategy(mesh, (3, 3)),),
            {"tensors": ts, "alpha": 1.0},
            RuntimeSchemaInfo(needs_pytree=True),
        )
        ks = schema.kwargs_strategy
        self.assertEqual(len(ks), 2)
        self.assertCountEqual(ks, (c0, c1))

    def test_expand_one_output_plus_args_and_kwarg_tensor_strategies(self):
        """Placement list: [output] + args + kwargs OpStrategies (same invariant as npu_grouped_matmul-style code)."""
        mesh = _cpu_mesh(self.ws)
        x_st = _replicate_strategy(mesh, (4, 4))
        bias_st = _replicate_strategy(mesh, (4, 4))
        # Positional tensors only in args_schema; bias only in kwargs (typical fused-op pattern).
        op_schema = OpSchema(
            torch.ops.aten.add.Tensor,
            (x_st,),
            {"bias": bias_st},
            RuntimeSchemaInfo(needs_pytree=False),
        )
        self.assertEqual(len(op_schema.args_strategy) + len(op_schema.kwargs_strategy), 2)
        # [y_out, x_in, bias_kw]
        single = [[Replicate(), Replicate(), Replicate()]]
        strat = expand_to_full_mesh_op_strategy(mesh, op_schema, single, input_index=1)
        self.assertGreater(len(strat.strategies), 0)
        self.assertEqual(len(strat.strategies[0].input_specs), 2)

    def test_expand_multi_output_index_with_kwargs(self):
        """Multiple leading outputs in placement list; kwargs still counted in input_args_strategy."""
        mesh = _cpu_mesh(self.ws)
        in0 = _replicate_strategy(mesh, (2, 8, 64, 128))
        in1 = _replicate_strategy(mesh, (2, 8, 64, 192))
        kw_st = _replicate_strategy(mesh, (1,))
        op_schema = OpSchema(
            torch.ops.aten.add.Tensor,
            (in0, in1),
            {"aux": kw_st},
            RuntimeSchemaInfo(needs_pytree=False),
        )
        # 2 outputs + 2 args + 1 kw = 5 placements; input_index=2 -> two output slots.
        single = [
            [
                Replicate(),
                Replicate(),
                Replicate(),
                Replicate(),
                Replicate(),
            ]
        ]
        strat = expand_to_full_mesh_op_strategy(mesh, op_schema, single, input_index=2)
        self.assertGreater(len(strat.strategies), 0)
        self.assertEqual(len(strat.strategies[0].input_specs), 3)


if __name__ == "__main__":
    run_tests()
