import numpy as np
import torch
from torch.distributed._tensor import distribute_tensor, Replicate, Shard
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize
)

import torch_npu
from torch_npu.testing.testcase import run_tests
from torch_npu.testing._internal.common_dtensor import NPUDTensorTestBase
from torch_npu.testing.common_distributed import with_comms, skipIfUnsupportMultiNPU
from torch_npu.testing.common_utils import SupportedDevices


def get_atten_mask(shape, sparse_mode=0, pre_tokens=65536, next_tokens=65536):
    atten_mask = None
    if sparse_mode == 0:
        atten_mask_u = np.triu(np.ones(shape), k=pre_tokens + 1)
        atten_mask_l = np.tril(np.ones(shape), k=-next_tokens - 1)
        atten_masks = atten_mask_u + atten_mask_l
        atten_mask = torch.tensor(atten_masks).to(torch.float16).bool().npu()
    elif sparse_mode in [2, 3, 4]:
        atten_masks = torch.from_numpy(np.triu(np.ones([2048, 2048]), k=1))
        atten_mask = torch.tensor(atten_masks).to(torch.float16).bool().npu()

    return atten_mask


class TestAttentionOps(NPUDTensorTestBase):
    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    @parametrize(
        "sparse_mode,pre_tokens,next_tokens",
        [
            (0, 128, 128),
            (1, 65536, 65536),
            (2, 65536, 0),
            (3, 65536, 0),
            (4, 128, 128)
        ]
    )
    def test_npu_fusion_attention_forward_bnsd(self, sparse_mode, pre_tokens, next_tokens):
        device_mesh = self.build_device_mesh()

        B, N, S, D = 4, 8, 32, 32
        shape = (B, N, S, D)
        query = torch.randn(shape, dtype=torch.float32, device="npu")
        key = torch.randn(shape, dtype=torch.float32, device="npu")
        value = torch.randn(shape, dtype=torch.float32, device="npu")

        scale = 0.08838

        atten_mask = get_atten_mask(shape, sparse_mode, pre_tokens, next_tokens)
        result = torch_npu.npu_fusion_attention(
            query, key, value, head_num=N, input_layout="BNSD", scale=scale, sparse_mode=sparse_mode,
            atten_mask=atten_mask, pre_tockens=pre_tokens, next_tockens=next_tokens
        )

        def test_placement_comb(query_placements, key_placements, value_placements, atten_mask_placements):
            dist_query = distribute_tensor(query, device_mesh, query_placements)
            dist_key = distribute_tensor(key, device_mesh, key_placements)
            dist_value = distribute_tensor(value, device_mesh, value_placements)
            dist_atten_mask = distribute_tensor(
                atten_mask, device_mesh, atten_mask_placements
            ) if atten_mask is not None else None
            dist_result = torch_npu.npu_fusion_attention(
                dist_query, dist_key, dist_value, head_num=N, input_layout="BNSD", scale=scale,
                sparse_mode=sparse_mode, atten_mask=dist_atten_mask,
                pre_tockens=pre_tokens, next_tockens=next_tokens
            )
            self.assertEqual(dist_result[0].full_tensor(), result[0])
            self.assertEqual(dist_result[1].full_tensor(), result[1])
            self.assertEqual(dist_result[2].full_tensor(), result[2])

        placements = [Shard(0), Shard(1), Shard(2), Shard(3), Replicate()]
        for placement in placements:
            if atten_mask is None or (isinstance(placement, Shard) and atten_mask.ndim <= placement.dim):
                test_placement_comb([placement], [placement], [placement], [Replicate()])
            else:
                test_placement_comb([placement], [placement], [placement], [placement])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    @parametrize(
        "sparse_mode,pre_tokens,next_tokens",
        [
            (0, 128, 128),
            (1, 65536, 65536),
            (2, 65536, 0),
            (3, 65536, 0),
            (4, 128, 128)
        ]
    )
    def test_npu_fusion_attention_backward_bnsd(self, sparse_mode, pre_tokens, next_tokens):
        device_mesh = self.build_device_mesh()

        B, N, S, D = 4, 8, 32, 32
        shape = (B, N, S, D)
        query = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)
        key = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)
        value = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)

        scale = 0.08838

        atten_mask = get_atten_mask(shape, sparse_mode, pre_tokens, next_tokens)
        result = torch_npu.npu_fusion_attention(
            query, key, value, head_num=N, input_layout="BNSD", scale=scale, sparse_mode=sparse_mode,
            atten_mask=atten_mask, pre_tockens=pre_tokens, next_tockens=next_tokens
        )
        gard_y = torch.ones_like(result[0])
        result[0].backward(gard_y)

        def test_placement_comb(query_placements, key_placements, value_placements, atten_mask_placements):
            dist_query = distribute_tensor(query, device_mesh, query_placements)
            dist_key = distribute_tensor(key, device_mesh, key_placements)
            dist_value = distribute_tensor(value, device_mesh, value_placements)
            dist_atten_mask = distribute_tensor(
                atten_mask, device_mesh, atten_mask_placements
            ) if atten_mask is not None else None
            dist_result = torch_npu.npu_fusion_attention(
                dist_query, dist_key, dist_value, head_num=N, input_layout="BNSD", scale=scale,
                sparse_mode=sparse_mode, atten_mask=dist_atten_mask,
                pre_tockens=pre_tokens, next_tockens=next_tokens
            )
            dist_grad_y = distribute_tensor(gard_y, device_mesh, dist_result[0].placements)
            dist_result[0].backward(dist_grad_y)
            self.assertEqual(dist_result[0].full_tensor(), result[0])
            self.assertEqual(dist_result[1].full_tensor(), result[1])
            self.assertEqual(dist_result[2].full_tensor(), result[2])
            self.assertEqual(dist_query.grad.full_tensor(), query.grad)
            self.assertEqual(dist_key.grad.full_tensor(), key.grad)
            self.assertEqual(dist_value.grad.full_tensor(), value.grad)

        placements = [Shard(0), Shard(1), Shard(2), Shard(3), Replicate()]
        for placement in placements:
            if atten_mask is None or (isinstance(placement, Shard) and atten_mask.ndim <= placement.dim):
                test_placement_comb([placement], [placement], [placement], [Replicate()])
            else:
                test_placement_comb([placement], [placement], [placement], [placement])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_npu_fusion_attention_bsnd(self):
        device_mesh = self.build_device_mesh()

        B, N, S, D = 4, 8, 32, 32
        shape = (B, S, N, D)
        query = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)
        key = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)
        value = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)
        scale = 0.08838

        result = torch_npu.npu_fusion_attention(query, key, value, head_num=N, input_layout="BSND", scale=scale)
        gard_y = torch.ones_like(result[0])
        result[0].backward(gard_y)

        def test_placement_comb(query_placements, key_placements, value_placements):
            dist_query = distribute_tensor(query, device_mesh, query_placements)
            dist_key = distribute_tensor(key, device_mesh, key_placements)
            dist_value = distribute_tensor(value, device_mesh, value_placements)
            dist_result = torch_npu.npu_fusion_attention(
                dist_query, dist_key, dist_value, head_num=N, input_layout="BSND", scale=scale
            )
            dist_grad_y = distribute_tensor(gard_y, device_mesh, dist_result[0].placements)
            dist_result[0].backward(dist_grad_y)
            self.assertEqual(dist_result[0].full_tensor(), result[0])
            self.assertEqual(dist_result[1].full_tensor(), result[1])
            self.assertEqual(dist_result[2].full_tensor(), result[2])
            self.assertEqual(dist_query.grad.full_tensor(), query.grad)
            self.assertEqual(dist_key.grad.full_tensor(), key.grad)
            self.assertEqual(dist_value.grad.full_tensor(), value.grad)

        placements = [Shard(0), Shard(1), Shard(2), Shard(3), Replicate()]
        for placement in placements:
            test_placement_comb([placement], [placement], [placement])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_npu_fusion_attention_bsh(self):
        device_mesh = self.build_device_mesh()

        B, N, S, D = 4, 8, 32, 32
        shape = (B, S, N * D)
        query = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)
        key = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)
        value = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)
        scale = 0.08838

        result = torch_npu.npu_fusion_attention(query, key, value, head_num=N, input_layout="BSH", scale=scale)
        gard_y = torch.ones_like(result[0])
        result[0].backward(gard_y)

        def test_placement_comb(query_placements, key_placements, value_placements):
            dist_query = distribute_tensor(query, device_mesh, query_placements)
            dist_key = distribute_tensor(key, device_mesh, key_placements)
            dist_value = distribute_tensor(value, device_mesh, value_placements)
            dist_result = torch_npu.npu_fusion_attention(
                dist_query, dist_key, dist_value, head_num=N, input_layout="BSH", scale=scale
            )
            dist_grad_y = distribute_tensor(gard_y, device_mesh, dist_result[0].placements)
            dist_result[0].backward(dist_grad_y)
            self.assertEqual(dist_result[0].full_tensor(), result[0])
            self.assertEqual(dist_result[1].full_tensor(), result[1])
            self.assertEqual(dist_result[2].full_tensor(), result[2])
            self.assertEqual(dist_query.grad.full_tensor(), query.grad)
            self.assertEqual(dist_key.grad.full_tensor(), key.grad)
            self.assertEqual(dist_value.grad.full_tensor(), value.grad)

        placements = [Shard(0), Shard(1), Shard(2), Replicate()]
        for placement in placements:
            test_placement_comb([placement], [placement], [placement])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_npu_fusion_attention_tnd(self):
        device_mesh = self.build_device_mesh()

        B, Nq, Nkv, D = 3, 8, 2, 32
        seq_qlen_list = np.array([1, 2, 3])
        actual_seq_qlen = np.cumsum(seq_qlen_list)
        Sq = seq_qlen_list.sum()
        seq_kvlen_list = np.array([3, 4, 5])
        actual_seq_kvlen = np.cumsum(seq_kvlen_list)
        Skv = seq_kvlen_list.sum()
        query = torch.randn(Sq, Nq, D, dtype=torch.float32, device="npu", requires_grad=True)
        key = torch.randn(Skv, Nkv, D, dtype=torch.float32, device="npu", requires_grad=True)
        value = torch.randn(Skv, Nkv, D, dtype=torch.float32, device="npu", requires_grad=True)
        scale = 1 / (D ** 0.5)

        result = torch_npu.npu_fusion_attention(
            query, key, value, head_num=Nq, input_layout="TND", scale=scale,
            actual_seq_qlen=actual_seq_qlen.tolist(), actual_seq_kvlen=actual_seq_kvlen.tolist(), softmax_layout="TND"
        )
        grad_y = torch.ones_like(result[0])
        result[0].backward(grad_y)

        def test_placement_comb(query_placements, key_placements, value_placements):
            dist_query = distribute_tensor(query, device_mesh, query_placements)
            dist_key = distribute_tensor(key, device_mesh, key_placements)
            dist_value = distribute_tensor(value, device_mesh, value_placements)
            dist_result = torch_npu.npu_fusion_attention(
                dist_query, dist_key, dist_value, head_num=Nq, input_layout="TND", scale=scale,
                actual_seq_qlen=actual_seq_qlen.tolist(), actual_seq_kvlen=actual_seq_kvlen.tolist(),
                softmax_layout="TND"
            )
            dist_grad_y = distribute_tensor(grad_y, device_mesh, dist_result[0].placements)
            dist_result[0].backward(dist_grad_y)
            self.assertEqual(dist_result[0].full_tensor(), result[0])
            self.assertEqual(dist_result[1].full_tensor(), result[1])
            self.assertEqual(dist_result[2].full_tensor(), result[2])
            self.assertEqual(dist_query.grad.full_tensor(), query.grad)
            self.assertEqual(dist_key.grad.full_tensor(), key.grad)
            self.assertEqual(dist_value.grad.full_tensor(), value.grad)

        placements = [Shard(0), Shard(1), Shard(2), Replicate()]
        for placement in placements:
            test_placement_comb([placement], [placement], [placement])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    @parametrize(
        "sparse_mode,pre_tokens,next_tokens",
        [
            (0, 128, 128),
            (1, 65536, 65536),
            (2, 65536, 0),
            (3, 65536, 0),
            (4, 128, 128)
        ]
    )
    def test_npu_fusion_attention_v3_forward_bnsd(self, sparse_mode, pre_tokens, next_tokens):
        device_mesh = self.build_device_mesh()

        B, N, S, D = 4, 8, 32, 32
        shape = (B, N, S, D)
        query = torch.randn(shape, dtype=torch.float32, device="npu")
        key = torch.randn(shape, dtype=torch.float32, device="npu")
        value = torch.randn(shape, dtype=torch.float32, device="npu")

        scale = 0.08838

        atten_mask = get_atten_mask(shape, sparse_mode, pre_tokens, next_tokens)
        result = torch_npu.npu_fusion_attention_v3(
            query, key, value, head_num=N, input_layout="BNSD", scale=scale, sparse_mode=sparse_mode,
            atten_mask=atten_mask, pre_tockens=pre_tokens, next_tockens=next_tokens
        )

        def test_placement_comb(query_placements, key_placements, value_placements, atten_mask_placements):
            dist_query = distribute_tensor(query, device_mesh, query_placements)
            dist_key = distribute_tensor(key, device_mesh, key_placements)
            dist_value = distribute_tensor(value, device_mesh, value_placements)
            dist_atten_mask = distribute_tensor(
                atten_mask, device_mesh, atten_mask_placements
            ) if atten_mask is not None else None
            dist_result = torch_npu.npu_fusion_attention_v3(
                dist_query, dist_key, dist_value, head_num=N, input_layout="BNSD", scale=scale,
                sparse_mode=sparse_mode, atten_mask=dist_atten_mask,
                pre_tockens=pre_tokens, next_tockens=next_tokens
            )
            self.assertEqual(dist_result[0].full_tensor(), result[0])
            self.assertEqual(dist_result[1].full_tensor(), result[1])
            self.assertEqual(dist_result[2].full_tensor(), result[2])

        placements = [Shard(0), Shard(1), Shard(2), Shard(3), Replicate()]
        for placement in placements:
            if atten_mask is None or (isinstance(placement, Shard) and atten_mask.ndim <= placement.dim):
                test_placement_comb([placement], [placement], [placement], [Replicate()])
            else:
                test_placement_comb([placement], [placement], [placement], [placement])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    @parametrize(
        "sparse_mode,pre_tokens,next_tokens",
        [
            (0, 128, 128),
            (1, 65536, 65536),
            (2, 65536, 0),
            (3, 65536, 0),
            (4, 128, 128)
        ]
    )
    def test_npu_fusion_attention_v3_backward_bnsd(self, sparse_mode, pre_tokens, next_tokens):
        device_mesh = self.build_device_mesh()

        B, N, S, D = 4, 8, 32, 32
        shape = (B, N, S, D)
        query = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)
        key = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)
        value = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)

        scale = 0.08838

        atten_mask = get_atten_mask(shape, sparse_mode, pre_tokens, next_tokens)
        result = torch_npu.npu_fusion_attention_v3(
            query, key, value, head_num=N, input_layout="BNSD", scale=scale, sparse_mode=sparse_mode,
            atten_mask=atten_mask, pre_tockens=pre_tokens, next_tockens=next_tokens
        )
        gard_y = torch.ones_like(result[0])
        result[0].backward(gard_y)

        def test_placement_comb(query_placements, key_placements, value_placements, atten_mask_placements):
            dist_query = distribute_tensor(query, device_mesh, query_placements)
            dist_key = distribute_tensor(key, device_mesh, key_placements)
            dist_value = distribute_tensor(value, device_mesh, value_placements)
            dist_atten_mask = distribute_tensor(
                atten_mask, device_mesh, atten_mask_placements
            ) if atten_mask is not None else None
            dist_result = torch_npu.npu_fusion_attention_v3(
                dist_query, dist_key, dist_value, head_num=N, input_layout="BNSD", scale=scale,
                sparse_mode=sparse_mode, atten_mask=dist_atten_mask,
                pre_tockens=pre_tokens, next_tockens=next_tokens
            )
            dist_grad_y = distribute_tensor(gard_y, device_mesh, dist_result[0].placements)
            dist_result[0].backward(dist_grad_y)
            self.assertEqual(dist_result[0].full_tensor(), result[0])
            self.assertEqual(dist_result[1].full_tensor(), result[1])
            self.assertEqual(dist_result[2].full_tensor(), result[2])
            self.assertEqual(dist_query.grad.full_tensor(), query.grad)
            self.assertEqual(dist_key.grad.full_tensor(), key.grad)
            self.assertEqual(dist_value.grad.full_tensor(), value.grad)

        placements = [Shard(0), Shard(1), Shard(2), Shard(3), Replicate()]
        for placement in placements:
            if atten_mask is None or (isinstance(placement, Shard) and atten_mask.ndim <= placement.dim):
                test_placement_comb([placement], [placement], [placement], [Replicate()])
            else:
                test_placement_comb([placement], [placement], [placement], [placement])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_npu_fusion_attention_v3_bsnd(self):
        device_mesh = self.build_device_mesh()

        B, N, S, D = 4, 8, 32, 32
        shape = (B, S, N, D)
        query = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)
        key = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)
        value = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)
        scale = 0.08838

        result = torch_npu.npu_fusion_attention_v3(query, key, value, head_num=N, input_layout="BSND", scale=scale)
        gard_y = torch.ones_like(result[0])
        result[0].backward(gard_y)

        def test_placement_comb(query_placements, key_placements, value_placements):
            dist_query = distribute_tensor(query, device_mesh, query_placements)
            dist_key = distribute_tensor(key, device_mesh, key_placements)
            dist_value = distribute_tensor(value, device_mesh, value_placements)
            dist_result = torch_npu.npu_fusion_attention_v3(
                dist_query, dist_key, dist_value, head_num=N, input_layout="BSND", scale=scale
            )
            dist_grad_y = distribute_tensor(gard_y, device_mesh, dist_result[0].placements)
            dist_result[0].backward(dist_grad_y)
            self.assertEqual(dist_result[0].full_tensor(), result[0])
            self.assertEqual(dist_result[1].full_tensor(), result[1])
            self.assertEqual(dist_result[2].full_tensor(), result[2])
            self.assertEqual(dist_query.grad.full_tensor(), query.grad)
            self.assertEqual(dist_key.grad.full_tensor(), key.grad)
            self.assertEqual(dist_value.grad.full_tensor(), value.grad)

        placements = [Shard(0), Shard(1), Shard(2), Shard(3), Replicate()]
        for placement in placements:
            test_placement_comb([placement], [placement], [placement])

    @SupportedDevices(['Ascend910B'])
    @skipIfUnsupportMultiNPU(2)
    @with_comms
    def test_npu_fusion_attention_v3_bsh(self):
        device_mesh = self.build_device_mesh()

        B, N, S, D = 4, 8, 32, 32
        shape = (B, S, N * D)
        query = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)
        key = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)
        value = torch.randn(shape, dtype=torch.float32, device="npu", requires_grad=True)
        scale = 0.08838

        result = torch_npu.npu_fusion_attention_v3(query, key, value, head_num=N, input_layout="BSH", scale=scale)
        gard_y = torch.ones_like(result[0])
        result[0].backward(gard_y)

        def test_placement_comb(query_placements, key_placements, value_placements):
            dist_query = distribute_tensor(query, device_mesh, query_placements)
            dist_key = distribute_tensor(key, device_mesh, key_placements)
            dist_value = distribute_tensor(value, device_mesh, value_placements)
            dist_result = torch_npu.npu_fusion_attention_v3(
                dist_query, dist_key, dist_value, head_num=N, input_layout="BSH", scale=scale
            )
            dist_grad_y = distribute_tensor(gard_y, device_mesh, dist_result[0].placements)
            dist_result[0].backward(dist_grad_y)
            self.assertEqual(dist_result[0].full_tensor(), result[0])
            self.assertEqual(dist_result[1].full_tensor(), result[1])
            self.assertEqual(dist_result[2].full_tensor(), result[2])
            self.assertEqual(dist_query.grad.full_tensor(), query.grad)
            self.assertEqual(dist_key.grad.full_tensor(), key.grad)
            self.assertEqual(dist_value.grad.full_tensor(), value.grad)

        placements = [Shard(0), Shard(1), Shard(2), Replicate()]
        for placement in placements:
            test_placement_comb([placement], [placement], [placement])


instantiate_parametrized_tests(TestAttentionOps)


if __name__ == "__main__":
    run_tests()