import itertools

import torch
from torch.distributed._tensor import distribute_tensor, Replicate, Shard
from torch.testing._internal.common_utils import DeterministicGuard
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase

import torch_npu
from torch_npu.testing.testcase import run_tests
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU


class TestAllGatherBaseMmOp(DTensorTestBase):
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

    @skipIfUnsupportMultiNPU(4)
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


class TestMmReduceScatterBaseOp(DTensorTestBase):
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

    @skipIfUnsupportMultiNPU(4)
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

    @skipIfUnsupportMultiNPU(4)
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
                bias_list.append(torch.randn(n, dtype=dtype, device="npu"))

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
                self.assertEqual(dist_output.full_tensor(), global_output.to(dtype), atol=0.05, rtol=0.05)
                self.assertEqual(dist_output.to_local(), output)

            placement = [Shard(0), Shard(1), Replicate()]
            placement_combs = itertools.product(placement, placement)
            for comb in placement_combs:
                bias_placement = Replicate() if comb[1] == Replicate() else Shard(0)
                test_placement_comb([comb[0]], [comb[1]], [bias_placement])

    @skipIfUnsupportMultiNPU(4)
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


if __name__ == "__main__":
    run_tests()
