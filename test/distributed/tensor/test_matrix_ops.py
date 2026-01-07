import itertools
import numpy as np

import torch
from torch.distributed._tensor import distribute_tensor, Replicate, Shard
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    instantiate_parametrized_tests,
    parametrize
)

import torch_npu
from torch_npu.testing.testcase import run_tests
from torch_npu.testing._internal.common_dtensor import NPUDTensorTestBase
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU
from torch_npu.testing.common_utils import SupportedDevices


class TestAllGatherBaseMmOp(NPUDTensorTestBase):
    def _get_global_tensor(self, x1_list, x2_list, bias_list=None, x1_scale_list=None, x2_scale_list=None):
        # Example:
        # x1_rank0 = [[1, 1]], x2_rank0 = [[2], [2]], x1_rank1 = [[3, 3]], x2_rank1 = [[4], [4]]
        # output_rank0 = [[1, 1], [3, 3]] @ [[2], [2]] = [[4], [12]]
        # output_rank1 = [[1, 1], [3, 3]] @ [[4], [4]] = [[8], [24]]
        # in global view: x1 = [[1, 1], [3, 3]], x2 = [[2, 4], [2, 4]]
        #                 output = x1 @ x2 = [[4, 8], [12, 24]]
        x1 = torch.cat(x1_list, dim=0)
        x2 = torch.cat(x2_list, dim=1)
        output = torch.mm(x1, x2)
        if bias_list is not None:
            bias = torch.cat(bias_list, dim=0)
            output += bias
        if x1_scale_list is not None and x2_scale_list is not None:
            x1_scale = torch.cat(x1_scale_list, dim=0)
            x2_scale = torch.cat(x2_scale_list, dim=1)
            output = x1_scale * x2_scale * output
        gather_out = x1

        return x1, x2, output, gather_out

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_npu_all_gather_base_mm(self):
        mesh = self.build_device_mesh()

        m, k, n = 8, 256, 16
        dtype = torch.bfloat16
        x1_list = []
        x2_list = []
        for _ in range(self.world_size):
            x1_list.append(torch.randn(m, k, dtype=dtype, device="npu"))
            x2_list.append(torch.randn(k, n, dtype=dtype, device="npu"))

        global_x1, global_x2, global_output, global_gather_out = self._get_global_tensor(x1_list, x2_list)

        group = mesh.get_group()
        if torch.__version__ > '2.0':
            hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(self.rank)
        else:
            hcom_name = group.get_hccl_comm_name(self.rank)

        output, gather_out = torch_npu.npu_all_gather_base_mm(
            x1_list[self.rank], x2_list[self.rank], hcom_name, self.world_size
        )

        def test_placement_comb(x1_placements, x2_placements):
            dist_x1 = distribute_tensor(global_x1, mesh, x1_placements)
            dist_x2 = distribute_tensor(global_x2, mesh, x2_placements)
            dist_output, dist_gather_out = torch_npu.npu_all_gather_base_mm(
                dist_x1, dist_x2, hcom_name, self.world_size
            )
            self.assertEqual(dist_output.full_tensor(), global_output.to(dtype))
            self.assertEqual(dist_gather_out.full_tensor(), global_gather_out.to(dtype))
            self.assertEqual(dist_output.redistribute(placements=[Shard(1)]).to_local(), output)
            self.assertEqual(dist_gather_out.to_local(), gather_out)

        placement = [Shard(0), Shard(1), Replicate()]
        placement_combs = itertools.product(placement, placement)
        for comb in placement_combs:
            test_placement_comb([comb[0]], [comb[1]])


class TestMmReduceScatterBaseOp(NPUDTensorTestBase):
    def _get_global_tensor(self, x1_list, x2_list, bias_list=None, x1_scale_list=None, x2_scale_list=None):
        # Example:
        # x1_rank0 = [[1], [1]], x2_rank0 = [[2, 2]], x1_rank1 = [[3], [3]], x2_rank1 = [[4, 4]]
        # reduce result = [[1], [1]] @ [[2, 2]] + [[3], [3]] @ [[4, 4]] = [[14, 14], [14, 14]]
        # output_rank0 = [[14, 14]], output_rank1 = [[14, 14]]
        # in global view: x1 = [[1, 3], [1, 3]], x2 = [[2, 2], [4, 4]]
        #                 output = x1 @ x2 = [[14, 14], [14, 14]]
        x1 = torch.cat(x1_list, dim=1)
        x2 = torch.cat(x2_list, dim=0)
        output = torch.zeros(x1.size(0), x2.size(1), device=x1.device)
        for i, (x1_, x2_) in enumerate(zip(x1_list, x2_list)):
            x1_, x2_ = x1_list[i], x2_list[i]
            if x1_.dtype == torch.int8:
                x1_ = x1_.to(torch.float32)
                x2_ = x2_.to(torch.float32)
            output_ = torch.mm(x1_, x2_)
            if bias_list is not None:
                output_ += bias_list[i]
            scale = 1.0
            if x1_scale_list is not None:
                scale = scale * x1_scale_list[i]
            if x2_scale_list is not None:
                scale = scale * x2_scale_list[i]
            output_ = scale * output_
            output += output_
        return x1, x2, output

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_npu_mm_reduce_scatter_base(self):
        with DeterministicGuard(True):
            mesh = self.build_device_mesh()

            m, k, n = 4, 256, 4
            dtype = torch.bfloat16
            x1_list = []
            x2_list = []
            for _ in range(self.world_size):
                x1_list.append(torch.randn(m, k, dtype=dtype, device="npu"))
                x2_list.append(torch.randn(k, n, dtype=dtype, device="npu"))

            global_x1, global_x2, global_output = self._get_global_tensor(x1_list, x2_list)

            group = mesh.get_group()
            if torch.__version__ > '2.0':
                hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(self.rank)
            else:
                hcom_name = group.get_hccl_comm_name(self.rank)

            output = torch_npu.npu_mm_reduce_scatter_base(
                x1_list[self.rank], x2_list[self.rank], hcom_name, self.world_size
            )

            def test_placement_comb(x1_placements, x2_placements):
                dist_x1 = distribute_tensor(global_x1, mesh, x1_placements)
                dist_x2 = distribute_tensor(global_x2, mesh, x2_placements)
                dist_output = torch_npu.npu_mm_reduce_scatter_base(dist_x1, dist_x2, hcom_name, self.world_size)
                # Reduce operation introduces numerical difference, the magnitude of the error depends on the dtype and
                # shape. As a reference, in gpu, mm between x1(shape[4, 256], bf16, S(1) on 4 dev) and x2(shape[256, 4],
                # bf16, S(0) on 4 dev) has more than 0.07 absolute error and relative error.
                self.assertEqual(dist_output.full_tensor(), global_output.to(dtype), atol=0.05, rtol=0.05)
                self.assertEqual(dist_output.to_local(), output)

            placement = [Shard(0), Shard(1), Replicate()]
            placement_combs = itertools.product(placement, placement)
            for comb in placement_combs:
                test_placement_comb([comb[0]], [comb[1]])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_npu_mm_reduce_scatter_base_bias(self):
        with DeterministicGuard(True):
            mesh = self.build_device_mesh()

            m, k, n = 4, 256, 4
            dtype = torch.bfloat16
            x1_list = []
            x2_list = []
            bias_list = []
            for _ in range(self.world_size):
                x1_list.append(torch.randn(m, k, dtype=dtype, device="npu"))
                x2_list.append(torch.randn(k, n, dtype=dtype, device="npu"))
                bias_list.append(torch.zeros(n, dtype=dtype, device="npu"))

            global_x1, global_x2, global_output = self._get_global_tensor(x1_list, x2_list, bias_list=bias_list)
            global_bias = torch.cat(bias_list, dim=0)

            group = mesh.get_group()
            if torch.__version__ > '2.0':
                hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(self.rank)
            else:
                hcom_name = group.get_hccl_comm_name(self.rank)

            output = torch_npu.npu_mm_reduce_scatter_base(
                x1_list[self.rank], x2_list[self.rank], hcom_name, self.world_size, bias=bias_list[self.rank]
            )

            def test_placement_comb(x1_placements, x2_placements, bias_placements):
                dist_x1 = distribute_tensor(global_x1, mesh, x1_placements)
                dist_x2 = distribute_tensor(global_x2, mesh, x2_placements)
                dist_bias = distribute_tensor(global_bias, mesh, bias_placements)
                dist_output = torch_npu.npu_mm_reduce_scatter_base(
                    dist_x1, dist_x2, hcom_name, self.world_size, bias=dist_bias
                )
                self.assertEqual(dist_output.full_tensor(), global_output.to(dtype))
                self.assertEqual(dist_output.to_local(), output)

            placement = [Shard(0), Shard(1), Replicate()]
            placement_combs = itertools.product(placement, placement)
            for comb in placement_combs:
                bias_placement = Replicate() if comb[1] == Replicate() else Shard(0)
                test_placement_comb([comb[0]], [comb[1]], [bias_placement])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_npu_mm_reduce_scatter_base_quant(self):
        with DeterministicGuard(True):
            mesh = self.build_device_mesh()

            m, k, n = 4, 256, 4
            dtype = torch.bfloat16
            x1_list = []
            x2_list = []
            x1_scale_list = []
            x2_scale_list = []
            for _ in range(self.world_size):
                x1_list.append(torch.randint(-10, 10, size=(m, k), dtype=torch.int8, device="npu"))
                x2_list.append(torch.randint(-10, 10, size=(k, n), dtype=torch.int8, device="npu"))
                x1_scale_list.append(torch.randn(m, 1, dtype=torch.float32, device="npu"))
                x2_scale_list.append(torch.randn(1, n, dtype=torch.float32, device="npu"))

            global_x1, global_x2, global_output = self._get_global_tensor(
                x1_list, x2_list, x1_scale_list=x1_scale_list, x2_scale_list=x2_scale_list
            )
            global_x1_scale = torch.cat(x1_scale_list, dim=1)
            global_x2_scale = torch.cat(x2_scale_list, dim=0)

            group = mesh.get_group()
            if torch.__version__ > '2.0':
                hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(self.rank)
            else:
                hcom_name = group.get_hccl_comm_name(self.rank)

            output = torch_npu.npu_mm_reduce_scatter_base(
                x1_list[self.rank], x2_list[self.rank], hcom_name, self.world_size,
                x1_scale=x1_scale_list[self.rank], x2_scale=x2_scale_list[self.rank],
                output_dtype=dtype, comm_mode="aiv"
            )

            def test_placement_comb(x1_placements, x2_placements, x1_scale_placements, x2_scale_placements):
                dist_x1 = distribute_tensor(global_x1, mesh, x1_placements)
                dist_x2 = distribute_tensor(global_x2, mesh, x2_placements)
                dist_x1_scale = distribute_tensor(global_x1_scale, mesh, x1_scale_placements)
                dist_x2_scale = distribute_tensor(global_x2_scale, mesh, x2_scale_placements)
                dist_output = torch_npu.npu_mm_reduce_scatter_base(
                    dist_x1, dist_x2, hcom_name, self.world_size,
                    x1_scale=dist_x1_scale, x2_scale=dist_x2_scale, output_dtype=dtype, comm_mode="aiv"
                )
                self.assertEqual(dist_output.full_tensor(), global_output.to(dtype))
                self.assertEqual(dist_output.to_local(), output)

            placement = [Shard(0), Shard(1), Replicate()]
            placement_combs = itertools.product(placement, placement)
            for comb in placement_combs:
                test_placement_comb([comb[0]], [comb[1]], [comb[0]], [comb[1]])


class TestGroupedMatMulOp(NPUDTensorTestBase):
    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    @parametrize("x_ndim", [2, 3])
    @parametrize("with_bias", [True, False])
    def test_npu_grouped_matmul_xNwNyN(self, x_ndim, with_bias):
        mesh = self.build_device_mesh()

        x_shapes = [(16, 16), (64, 16), (32, 64)] if x_ndim == 2 else [(7, 16, 16), (8, 64, 16), (9, 32, 64)]
        x = [torch.randn(shape, dtype=torch.float32, device="npu") for shape in x_shapes]
        weight_shapes = [(16, 16), (16, 64), (64, 8)]
        weight, bias = [], []
        for shape in weight_shapes:
            weight.append(torch.randn(shape, dtype=torch.float32, device="npu"))
            if with_bias:
                bias.append(torch.randn(shape[1], dtype=torch.float32, device="npu"))
        group_list = None
        split_item = 0
        group_type = -1

        y = torch_npu.npu_grouped_matmul(
            x, weight, bias=bias, group_list=group_list, split_item=split_item, group_type=group_type
        )

        def test_placement_comb(x_placements, weight_placements, bias_placements):
            dist_x = [distribute_tensor(x_i, mesh, x_placements) for x_i in x]
            dist_weight = [distribute_tensor(weight_i, mesh, weight_placements) for weight_i in weight]
            dist_bias = [distribute_tensor(bias_i, mesh, bias_placements) for bias_i in bias]
            dist_y = torch_npu.npu_grouped_matmul(
                dist_x, dist_weight, bias=dist_bias, group_list=group_list,
                split_item=split_item, group_type=group_type
            )
            for dist_y_i, y_i in zip(dist_y, y):
                self.assertEqual(dist_y_i.full_tensor(), y_i)

        placement = [Shard(0), Shard(1), Replicate()]
        placement_combs = itertools.product(placement, placement, [Shard(0), Replicate()])
        for comb in placement_combs:
            test_placement_comb([comb[0]], [comb[1]], [comb[2]])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    @parametrize("with_bias", [True, False])
    def test_npu_grouped_matmul_x1w1y1(self, with_bias):
        mesh = self.build_device_mesh()

        x = [torch.randn(8, 8, dtype=torch.float16, device="npu")]
        weight = [torch.randn(2, 8, 8, dtype=torch.float16, device="npu")]
        bias = [torch.randn(2, 8, dtype=torch.float16, device="npu")] if with_bias else None
        group_list = torch.tensor([2, 8], device="npu")
        split_item = 3
        group_type = 0

        y = torch_npu.npu_grouped_matmul(
            x, weight, bias=bias, group_list=group_list, split_item=split_item, group_type=group_type
        )

        def test_placement_comb(x_placements, weight_placements, bias_placements, group_list_placements):
            dist_x = [distribute_tensor(x_i, mesh, x_placements) for x_i in x]
            dist_weight = [distribute_tensor(weight_i, mesh, weight_placements) for weight_i in weight]
            dist_bias = [distribute_tensor(bias_i, mesh, bias_placements) for bias_i in bias] if bias else None
            dist_group_list = distribute_tensor(group_list, mesh, group_list_placements)
            dist_y = torch_npu.npu_grouped_matmul(
                dist_x, dist_weight, bias=dist_bias, group_list=dist_group_list,
                split_item=split_item, group_type=group_type
            )
            for dist_y_i, y_i in zip(dist_y, y):
                self.assertEqual(dist_y_i.full_tensor(), y_i, atol=0.001, rtol=0.02)

        placement = [Shard(0), Shard(1), Replicate()]
        placement_combs = itertools.product(placement, placement, [Shard(0), Replicate()])
        for comb in placement_combs:
            test_placement_comb([comb[0]], [comb[1]], [comb[2]], [comb[2]])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    @parametrize("with_bias", [True, False])
    def test_npu_grouped_matmul_xNwNy1(self, with_bias):
        mesh = self.build_device_mesh()

        x_shapes = [(16, 16), (64, 16), (32, 64)]
        x = [torch.randn(shape, dtype=torch.float32, device="npu") for shape in x_shapes]
        weight_shapes = [(16, 16), (16, 16), (64, 16)]
        weight, bias = [], []
        for shape in weight_shapes:
            weight.append(torch.randn(shape, dtype=torch.float32, device="npu"))
            if with_bias:
                bias.append(torch.randn(shape[1], dtype=torch.float32, device="npu"))
        group_list = None
        split_item = 2
        group_type = 0

        y = torch_npu.npu_grouped_matmul(
            x, weight, bias=bias, group_list=group_list, split_item=split_item, group_type=group_type
        )

        def test_placement_comb(x_placements, weight_placements, bias_placements):
            dist_x = [distribute_tensor(x_i, mesh, x_placements) for x_i in x]
            dist_weight = [distribute_tensor(weight_i, mesh, weight_placements) for weight_i in weight]
            dist_bias = [distribute_tensor(bias_i, mesh, bias_placements) for bias_i in bias]
            dist_y = torch_npu.npu_grouped_matmul(
                dist_x, dist_weight, bias=dist_bias, group_list=group_list,
                split_item=split_item, group_type=group_type
            )
            for dist_y_i, y_i in zip(dist_y, y):
                self.assertEqual(dist_y_i.full_tensor(), y_i)

        placement = [Shard(0), Shard(1), Replicate()]
        placement_combs = itertools.product(placement, placement, [Shard(0), Replicate()])
        for comb in placement_combs:
            test_placement_comb([comb[0]], [comb[1]], [comb[2]])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    @parametrize("with_bias", [True, False])
    @parametrize("group_type", [None, 0])
    def test_npu_grouped_matmul_x1wNy1(self, with_bias, group_type):
        mesh = self.build_device_mesh()

        x = [torch.randn(8, 8, dtype=torch.float16, device="npu")]
        weight = [torch.randn(8, 4, dtype=torch.float16, device="npu") for _ in range(2)]
        bias = [torch.randn(4, dtype=torch.float16, device="npu") for _ in range(2)] if with_bias else None
        group_list = [4, 8]
        if group_type is not None:
            group_list = torch.tensor(group_list).npu()
        split_item = 3

        y = torch_npu.npu_grouped_matmul(
            x, weight, bias=bias, group_list=group_list, split_item=split_item, group_type=group_type
        )

        def test_placement_comb(x_placements, weight_placements, bias_placements, group_list_placements):
            dist_x = [distribute_tensor(x_i, mesh, x_placements) for x_i in x]
            dist_weight = [distribute_tensor(weight_i, mesh, weight_placements) for weight_i in weight]
            dist_bias = [distribute_tensor(bias_i, mesh, bias_placements) for bias_i in bias] if bias else None
            if group_type is not None:
                dist_group_list = distribute_tensor(group_list, mesh, group_list_placements)
                dist_y = torch_npu.npu_grouped_matmul(
                    dist_x, dist_weight, bias=dist_bias, group_list=dist_group_list,
                    split_item=split_item, group_type=group_type
                )
            else:
                dist_y = torch_npu.npu_grouped_matmul(
                    dist_x, dist_weight, bias=dist_bias, group_list=group_list,
                    split_item=split_item
                )
            for dist_y_i, y_i in zip(dist_y, y):
                self.assertEqual(dist_y_i.full_tensor(), y_i, atol=0.001, rtol=0.02)

        placement = [Shard(0), Shard(1), Replicate()]
        placement_combs = itertools.product(placement, placement, [Shard(0), Replicate()])
        for comb in placement_combs:
            test_placement_comb([comb[0]], [comb[1]], [comb[2]], [comb[2]])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    @parametrize("with_bias", [True, False])
    def test_npu_grouped_matmul_x1wNyN(self, with_bias):
        mesh = self.build_device_mesh()

        x = [torch.randn(16, 8, dtype=torch.float16, device="npu")]
        weight_shapes = [(8, 4), (8, 4)]
        weight, bias = [], []
        for shape in weight_shapes:
            weight.append(torch.randn(shape, dtype=torch.float16, device="npu"))
            if with_bias:
                bias.append(torch.randn(shape[1], dtype=torch.float16, device="npu"))
        group_list = [8, 16]
        split_item = 1

        y = torch_npu.npu_grouped_matmul(x, weight, bias=bias, group_list=group_list, split_item=split_item)

        def test_placement_comb(x_placements, weight_placements, bias_placements):
            dist_x = [distribute_tensor(x_i, mesh, x_placements) for x_i in x]
            dist_weight = [distribute_tensor(weight_i, mesh, weight_placements) for weight_i in weight]
            dist_bias = [distribute_tensor(bias_i, mesh, bias_placements) for bias_i in bias]
            dist_y = torch_npu.npu_grouped_matmul(
                dist_x, dist_weight, bias=dist_bias, group_list=group_list, split_item=split_item
            )
            for dist_y_i, y_i in zip(dist_y, y):
                self.assertEqual(dist_y_i.full_tensor(), y_i, atol=0.001, rtol=0.02)

        placement = [Shard(0), Shard(1), Replicate()]
        placement_combs = itertools.product(placement, placement, [Shard(0), Replicate()])
        for comb in placement_combs:
            test_placement_comb([comb[0]], [comb[1]], [comb[2]])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_npu_grouped_matmul_quant(self):
        mesh = self.build_device_mesh()

        x = [torch.randint(1, 5, size=(112, 64), dtype=torch.int8, device="npu")]
        weight = [torch.randint(1, 5, size=(64, 16), dtype=torch.int8, device="npu") for _ in range(3)]
        bias = [torch.randint(1, 5, size=(16,), dtype=torch.int32, device="npu") for _ in range(3)]
        scale = [torch.randn(16, dtype=torch.float32, device="npu") for _ in range(3)]
        group_list = torch.tensor([32, 80, 112], device="npu")
        split_item = 3
        group_type = 0

        y = torch_npu.npu_grouped_matmul(
            x, weight, bias=bias, scale=scale, group_list=group_list, split_item=split_item, group_type=group_type,
            output_dtype=torch.float16
        )

        def test_placement_comb(x_placements, weight_placements, bias_placements, group_list_placements):
            dist_x = [distribute_tensor(x_i, mesh, x_placements) for x_i in x]
            dist_weight = [distribute_tensor(weight_i, mesh, weight_placements) for weight_i in weight]
            dist_bias = [distribute_tensor(bias_i, mesh, bias_placements) for bias_i in bias]
            dist_scale = [distribute_tensor(scale_i, mesh, bias_placements) for scale_i in scale]
            dist_group_list = distribute_tensor(group_list, mesh, group_list_placements)
            dist_y = torch_npu.npu_grouped_matmul(
                dist_x, dist_weight, bias=dist_bias, scale=dist_scale, group_list=dist_group_list,
                split_item=split_item, group_type=group_type, output_dtype=torch.float16
            )
            for dist_y_i, y_i in zip(dist_y, y):
                self.assertEqual(dist_y_i.full_tensor(), y_i)

        placement = [Shard(0), Shard(1), Replicate()]
        placement_combs = itertools.product(placement, placement, [Shard(0), Replicate()])
        for comb in placement_combs:
            test_placement_comb([comb[0]], [comb[1]], [comb[2]], [comb[2]])


class TestApplyAdamW(NPUDTensorTestBase):
    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_apply_adam_w_replicate(self):
        mesh = self.build_device_mesh()

        amsgrad = False
        maximize = True
        scalar_shape = [1]
        input_size = (21130, 512)

        var_npu = torch.randn(input_size, device="npu")
        m_npu = torch.randn(input_size, device="npu")
        v_npu = torch.randn(input_size, device="npu")
        grad_npu = torch.randn(input_size, device="npu")

        var_npu_dtensor = distribute_tensor(var_npu, mesh, [Replicate()])
        m_npu_dtensor = distribute_tensor(m_npu, mesh, [Replicate()])
        v_npu_dtensor = distribute_tensor(v_npu, mesh, [Replicate()])
        grad_npu_dtensor = distribute_tensor(grad_npu, mesh, [Replicate()])
        
        np.random.seed(42)

        beta1_power = np.random.uniform(0.0, 1.0, scalar_shape)
        beta2_power = np.random.uniform(0.0, 1.0, scalar_shape)
        lr = np.random.uniform(0.0001, 0.1, scalar_shape)
        weight_decay = np.random.uniform(0.001, 0.1, scalar_shape)
        beta1 = np.random.uniform(0.5, 1.0, scalar_shape)
        beta2 = np.random.uniform(0.5, 1.0, scalar_shape)
        eps = np.random.uniform(0.00001, 0.01, scalar_shape)
        max_grad_norm = None
        
        var_ret_npu, m_ret_npu, v_ret_npu = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu, m_npu, v_npu),
        )

        var_ret_npu_dtensor, m_ret_npu_dtensor, v_ret_npu_dtensor = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu_dtensor,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu_dtensor, m_npu_dtensor, v_npu_dtensor),
        )

        self.assertEqual(var_ret_npu_dtensor.full_tensor(), var_ret_npu)
        self.assertEqual(m_ret_npu_dtensor.full_tensor(), m_ret_npu)
        self.assertEqual(v_ret_npu_dtensor.full_tensor(), v_ret_npu)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_apply_adam_w_shard00(self):
        mesh = self.build_device_mesh()

        amsgrad = False
        maximize = True
        scalar_shape = [1]
        input_size = (21130, 512)

        var_npu = torch.randn(input_size, device="npu")
        m_npu = torch.randn(input_size, device="npu")
        v_npu = torch.randn(input_size, device="npu")
        grad_npu = torch.randn(input_size, device="npu")

        var_npu_dtensor = distribute_tensor(var_npu, mesh, [Shard(0)])
        m_npu_dtensor = distribute_tensor(m_npu, mesh, [Shard(0)])
        v_npu_dtensor = distribute_tensor(v_npu, mesh, [Shard(0)])
        grad_npu_dtensor = distribute_tensor(grad_npu, mesh, [Shard(0)])
        
        np.random.seed(42)

        beta1_power = np.random.uniform(0.0, 1.0, scalar_shape)
        beta2_power = np.random.uniform(0.0, 1.0, scalar_shape)
        lr = np.random.uniform(0.0001, 0.1, scalar_shape)
        weight_decay = np.random.uniform(0.001, 0.1, scalar_shape)
        beta1 = np.random.uniform(0.5, 1.0, scalar_shape)
        beta2 = np.random.uniform(0.5, 1.0, scalar_shape)
        eps = np.random.uniform(0.00001, 0.01, scalar_shape)
        max_grad_norm = None
        
        var_ret_npu, m_ret_npu, v_ret_npu = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu, m_npu, v_npu),
        )

        var_ret_npu_dtensor, m_ret_npu_dtensor, v_ret_npu_dtensor = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu_dtensor,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu_dtensor, m_npu_dtensor, v_npu_dtensor),
        )

        self.assertEqual(var_ret_npu_dtensor.full_tensor(), var_ret_npu)
        self.assertEqual(m_ret_npu_dtensor.full_tensor(), m_ret_npu)
        self.assertEqual(v_ret_npu_dtensor.full_tensor(), v_ret_npu)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_apply_adam_w_shard01(self):
        mesh = self.build_device_mesh()

        amsgrad = False
        maximize = True
        scalar_shape = [1]
        input_size = (21130, 512)

        var_npu = torch.randn(input_size, device="npu")
        m_npu = torch.randn(input_size, device="npu")
        v_npu = torch.randn(input_size, device="npu")
        grad_npu = torch.randn(input_size, device="npu")

        var_npu_dtensor = distribute_tensor(var_npu, mesh, [Shard(0)])
        m_npu_dtensor = distribute_tensor(m_npu, mesh, [Shard(0)])
        v_npu_dtensor = distribute_tensor(v_npu, mesh, [Shard(0)])
        grad_npu_dtensor = distribute_tensor(grad_npu, mesh, [Shard(1)])
        
        np.random.seed(42)

        beta1_power = np.random.uniform(0.0, 1.0, scalar_shape)
        beta2_power = np.random.uniform(0.0, 1.0, scalar_shape)
        lr = np.random.uniform(0.0001, 0.1, scalar_shape)
        weight_decay = np.random.uniform(0.001, 0.1, scalar_shape)
        beta1 = np.random.uniform(0.5, 1.0, scalar_shape)
        beta2 = np.random.uniform(0.5, 1.0, scalar_shape)
        eps = np.random.uniform(0.00001, 0.01, scalar_shape)
        max_grad_norm = None
        
        var_ret_npu, m_ret_npu, v_ret_npu = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu, m_npu, v_npu),
        )

        var_ret_npu_dtensor, m_ret_npu_dtensor, v_ret_npu_dtensor = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu_dtensor,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu_dtensor, m_npu_dtensor, v_npu_dtensor),
        )

        self.assertEqual(var_ret_npu_dtensor.full_tensor(), var_ret_npu)
        self.assertEqual(m_ret_npu_dtensor.full_tensor(), m_ret_npu)
        self.assertEqual(v_ret_npu_dtensor.full_tensor(), v_ret_npu)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_apply_adam_w_shard10(self):
        mesh = self.build_device_mesh()

        amsgrad = False
        maximize = True
        scalar_shape = [1]
        input_size = (21130, 512)

        var_npu = torch.randn(input_size, device="npu")
        m_npu = torch.randn(input_size, device="npu")
        v_npu = torch.randn(input_size, device="npu")
        grad_npu = torch.randn(input_size, device="npu")

        var_npu_dtensor = distribute_tensor(var_npu, mesh, [Shard(1)])
        m_npu_dtensor = distribute_tensor(m_npu, mesh, [Shard(1)])
        v_npu_dtensor = distribute_tensor(v_npu, mesh, [Shard(1)])
        grad_npu_dtensor = distribute_tensor(grad_npu, mesh, [Shard(0)])
        
        np.random.seed(42)

        beta1_power = np.random.uniform(0.0, 1.0, scalar_shape)
        beta2_power = np.random.uniform(0.0, 1.0, scalar_shape)
        lr = np.random.uniform(0.0001, 0.1, scalar_shape)
        weight_decay = np.random.uniform(0.001, 0.1, scalar_shape)
        beta1 = np.random.uniform(0.5, 1.0, scalar_shape)
        beta2 = np.random.uniform(0.5, 1.0, scalar_shape)
        eps = np.random.uniform(0.00001, 0.01, scalar_shape)
        max_grad_norm = None
        
        var_ret_npu, m_ret_npu, v_ret_npu = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu, m_npu, v_npu),
        )

        var_ret_npu_dtensor, m_ret_npu_dtensor, v_ret_npu_dtensor = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu_dtensor,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu_dtensor, m_npu_dtensor, v_npu_dtensor),
        )

        self.assertEqual(var_ret_npu_dtensor.full_tensor(), var_ret_npu)
        self.assertEqual(m_ret_npu_dtensor.full_tensor(), m_ret_npu)
        self.assertEqual(v_ret_npu_dtensor.full_tensor(), v_ret_npu)

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_torch_npu_npu_apply_adam_w_shard11(self):
        mesh = self.build_device_mesh()

        amsgrad = False
        maximize = True
        scalar_shape = [1]
        input_size = (21130, 512)

        var_npu = torch.randn(input_size, device="npu")
        m_npu = torch.randn(input_size, device="npu")
        v_npu = torch.randn(input_size, device="npu")
        grad_npu = torch.randn(input_size, device="npu")

        var_npu_dtensor = distribute_tensor(var_npu, mesh, [Shard(1)])
        m_npu_dtensor = distribute_tensor(m_npu, mesh, [Shard(1)])
        v_npu_dtensor = distribute_tensor(v_npu, mesh, [Shard(1)])
        grad_npu_dtensor = distribute_tensor(grad_npu, mesh, [Shard(1)])
        
        np.random.seed(42)

        beta1_power = np.random.uniform(0.0, 1.0, scalar_shape)
        beta2_power = np.random.uniform(0.0, 1.0, scalar_shape)
        lr = np.random.uniform(0.0001, 0.1, scalar_shape)
        weight_decay = np.random.uniform(0.001, 0.1, scalar_shape)
        beta1 = np.random.uniform(0.5, 1.0, scalar_shape)
        beta2 = np.random.uniform(0.5, 1.0, scalar_shape)
        eps = np.random.uniform(0.00001, 0.01, scalar_shape)
        max_grad_norm = None
        
        var_ret_npu, m_ret_npu, v_ret_npu = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu, m_npu, v_npu),
        )

        var_ret_npu_dtensor, m_ret_npu_dtensor, v_ret_npu_dtensor = torch_npu.npu_apply_adam_w(
            beta1_power[0],
            beta2_power[0],
            lr[0],
            weight_decay[0],
            beta1[0],
            beta2[0],
            eps[0],
            grad_npu_dtensor,
            max_grad_norm,
            amsgrad,
            maximize,
            out=(var_npu_dtensor, m_npu_dtensor, v_npu_dtensor),
        )

        self.assertEqual(var_ret_npu_dtensor.full_tensor(), var_ret_npu)
        self.assertEqual(m_ret_npu_dtensor.full_tensor(), m_ret_npu)
        self.assertEqual(v_ret_npu_dtensor.full_tensor(), v_ret_npu)


instantiate_parametrized_tests(TestGroupedMatMulOp)


if __name__ == "__main__":
    run_tests()
