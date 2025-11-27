import itertools

import torch
from torch.distributed._tensor import distribute_tensor, Replicate, Shard

import torch_npu
from torch_npu.testing.testcase import run_tests
from torch_npu.testing._internal.common_dtensor import NPUDTensorTestBase
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU


class TestMoeOps(NPUDTensorTestBase):
    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_npu_moe_token_permute_forward(self):
        device_mesh = self.build_device_mesh()

        num_tokens = 16
        hidden_size = 8
        topk = 4
        tokens = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="npu")
        indices = torch.randint(0, 4, (num_tokens, topk), dtype=torch.int32, device="npu")

        permuted_tokens, sorted_indices = torch_npu.npu_moe_token_permute(tokens, indices)

        def test_placement_comb(placements1, placements2):
            dist_tokens = distribute_tensor(tokens, device_mesh, placements1)
            dist_indices = distribute_tensor(indices, device_mesh, placements2)
            dist_permuted_tokens, dist_sorted_indices = torch_npu.npu_moe_token_permute(dist_tokens, dist_indices)
            self.assertEqual(dist_permuted_tokens.full_tensor(), permuted_tokens)
            self.assertEqual(dist_sorted_indices.full_tensor(), sorted_indices)

        placement = [Shard(0), Shard(1), Replicate()]
        placement_combs = itertools.product(placement, placement)
        for comb in placement_combs:
            test_placement_comb([comb[0]], [comb[1]])

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_npu_moe_token_permute_backward(self):
        device_mesh = self.build_device_mesh()

        num_tokens = 16
        hidden_size = 8
        topk = 4
        tokens = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="npu", requires_grad=True)
        indices = torch.randint(0, 4, (num_tokens, topk), dtype=torch.int32, device="npu")

        permuted_tokens, sorted_indices = torch_npu.npu_moe_token_permute(tokens, indices)
        grad_permuted_tokens = torch.ones_like(permuted_tokens, dtype=torch.bfloat16, device="npu")
        permuted_tokens.backward(grad_permuted_tokens)

        def test_placement_comb(placements1, placements2):
            dist_tokens = distribute_tensor(tokens, device_mesh, placements1)
            dist_indices = distribute_tensor(indices, device_mesh, placements2)
            dist_permuted_tokens, dist_sorted_indices = torch_npu.npu_moe_token_permute(dist_tokens, dist_indices)
            dist_grad_permuted_tokens = distribute_tensor(
                grad_permuted_tokens, device_mesh, dist_permuted_tokens.placements
            )
            dist_permuted_tokens.backward(dist_grad_permuted_tokens)
            self.assertEqual(dist_permuted_tokens.full_tensor(), permuted_tokens)
            self.assertEqual(dist_sorted_indices.full_tensor(), sorted_indices)
            self.assertEqual(dist_tokens.grad.full_tensor(), tokens.grad)

        placement = [Shard(0), Shard(1), Replicate()]
        placement_combs = itertools.product(placement, placement)
        for comb in placement_combs:
            test_placement_comb([comb[0]], [comb[1]])

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_npu_moe_token_permute_clip(self):
        device_mesh = self.build_device_mesh()

        num_tokens = 16
        hidden_size = 8
        topk = 4
        num_out_tokens = 10
        tokens = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="npu", requires_grad=True)
        indices = torch.randint(0, 4, (num_tokens, topk), dtype=torch.int32, device="npu")

        permuted_tokens, sorted_indices = torch_npu.npu_moe_token_permute(
            tokens, indices, num_out_tokens=num_out_tokens
        )
        permuted_tokens.sum().backward()

        dist_tokens = distribute_tensor(tokens, device_mesh, [Shard(1)])
        dist_indices = distribute_tensor(indices, device_mesh, [Replicate()])
        dist_permuted_tokens, dist_sorted_indices = torch_npu.npu_moe_token_permute(
            dist_tokens, dist_indices, num_out_tokens=num_out_tokens
        )
        dist_permuted_tokens.sum().backward()

        self.assertEqual(dist_permuted_tokens.full_tensor(), permuted_tokens)
        self.assertEqual(dist_sorted_indices.full_tensor(), sorted_indices)
        self.assertEqual(dist_tokens.grad.full_tensor(), tokens.grad)

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_npu_moe_token_unpermute_forward(self):
        device_mesh = self.build_device_mesh()

        num_tokens = 16
        hidden_size = 8
        topk = 4
        permuted_tokens = torch.randn(num_tokens * topk, hidden_size, dtype=torch.bfloat16, device="npu")
        sorted_indices = torch.randperm(num_tokens * topk, dtype=torch.int32, device="npu")

        tokens = torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices)

        def test_placement_comb(placements1, placements2):
            dist_permuted_tokens = distribute_tensor(permuted_tokens, device_mesh, placements1)
            dist_sorted_indices = distribute_tensor(sorted_indices, device_mesh, placements2)
            dist_tokens = torch_npu.npu_moe_token_unpermute(dist_permuted_tokens, dist_sorted_indices)
            self.assertEqual(dist_tokens.full_tensor(), tokens)

        permuted_tokens_placement = [Shard(0), Shard(1), Replicate()]
        sorted_indices_placement = [Shard(0), Replicate()]
        placement_combs = itertools.product(permuted_tokens_placement, sorted_indices_placement)
        for comb in placement_combs:
            test_placement_comb([comb[0]], [comb[1]])

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_npu_moe_token_unpermute_backward(self):
        device_mesh = self.build_device_mesh()

        num_tokens = 8
        hidden_size = 4
        topk = 2
        permuted_tokens = torch.randn(
            num_tokens * topk, hidden_size, dtype=torch.bfloat16, device="npu", requires_grad=True
        )
        sorted_indices = torch.randperm(num_tokens * topk, dtype=torch.int32, device="npu")
        probs = torch.randn(num_tokens, topk, dtype=torch.bfloat16, device="npu", requires_grad=True)

        tokens = torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs)
        grad_tokens = torch.ones_like(tokens, dtype=torch.bfloat16, device="npu")
        tokens.backward(grad_tokens)

        def test_placement_comb(placements1, placements2):
            dist_permuted_tokens = distribute_tensor(permuted_tokens, device_mesh, placements1)
            dist_sorted_indices = distribute_tensor(sorted_indices, device_mesh, placements2)
            dist_probs = distribute_tensor(probs, device_mesh, [Shard(0)])
            dist_tokens = torch_npu.npu_moe_token_unpermute(dist_permuted_tokens, dist_sorted_indices, dist_probs)
            dist_grad_tokens = distribute_tensor(
                grad_tokens, device_mesh, dist_tokens.placements
            )
            dist_tokens.backward(dist_grad_tokens)
            self.assertEqual(dist_tokens.full_tensor(), tokens)
            self.assertEqual(dist_permuted_tokens.grad.full_tensor(), permuted_tokens.grad)
            self.assertEqual(dist_probs.grad.full_tensor(), probs.grad)

        permuted_tokens_placement = [Shard(0), Shard(1), Replicate()]
        sorted_indices_placement = [Shard(0), Replicate()]
        placement_combs = itertools.product(permuted_tokens_placement, sorted_indices_placement)
        for comb in placement_combs:
            test_placement_comb([comb[0]], [comb[1]])

    @skipIfUnsupportMultiNPU(4)
    @with_comms
    def test_npu_moe_token_permute_unpermute(self):
        device_mesh = self.build_device_mesh()

        num_tokens = 16
        hidden_size = 8
        topk = 4
        tokens = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="npu")
        indices = torch.randint(0, 4, (num_tokens, topk), dtype=torch.int32, device="npu")

        permuted_tokens, sorted_indices = torch_npu.npu_moe_token_permute(tokens, indices)
        reconstruct_tokens = torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices)

        dist_tokens = distribute_tensor(tokens, device_mesh, [Shard(1)])
        dist_indices = distribute_tensor(indices, device_mesh, [Replicate()])
        dist_permuted_tokens, dist_sorted_indices = torch_npu.npu_moe_token_permute(dist_tokens, dist_indices)
        dist_reconstruct_tokens = torch_npu.npu_moe_token_unpermute(dist_permuted_tokens, dist_sorted_indices)

        self.assertEqual(dist_permuted_tokens.full_tensor(), permuted_tokens)
        self.assertEqual(dist_sorted_indices.full_tensor(), sorted_indices)
        self.assertEqual(dist_reconstruct_tokens.full_tensor(), reconstruct_tokens)


if __name__ == "__main__":
    run_tests()