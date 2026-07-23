import torch
from torch.distributed._tensor import Partial, Replicate, Shard
from torch.distributed._tensor.experimental import register_sharding

npu = torch.ops.npu


@register_sharding(npu.npu_moe_token_permute.default)
def npu_moe_token_permute_strategy(tokens, indices, num_out_tokens=None, padded_mode=False):
    # func: npu_moe_token_permute(Tensor tokens, Tensor indices, int? num_out_tokens=None, bool padded_mode=False)
    #                            -> (Tensor, Tensor)
    strategies = []

    # all replicate strategy
    replicate_strategy = (
        [Replicate(), Replicate()], # output
        [Replicate(), Replicate(), None, None] # input
    )
    strategies.append(replicate_strategy)

    # hidden_size dim sharding strategy
    hidden_size_sharding_strategy = (
        [Shard(1), Replicate()],
        [Shard(1), Replicate(), None, None]
    )
    strategies.append(hidden_size_sharding_strategy)

    return strategies


@register_sharding(npu.npu_moe_token_permute_grad.default)
def npu_moe_token_permute_grad_strategy(tokens, grad_permuted_tokens, indices, sorted_indices, padded_mode=False):
    # func: npu_moe_token_permute_grad(Tensor tokens, Tensor grad_permuted_tokens, Tensor indices,
    #                                  Tensor sorted_indices, bool padded_mode=False) -> Tensor
    strategies = []

    # all replicate strategy
    replicate_strategy = (
        [Replicate()], # output
        [Replicate(), Replicate(), Replicate(), Replicate(), None] # input
    )
    strategies.append(replicate_strategy)

    # hidden_size dim sharding strategy
    hidden_size_sharding_strategy = (
        [Shard(1)],
        [Shard(1), Shard(1), Replicate(), Replicate(), None]
    )
    strategies.append(hidden_size_sharding_strategy)

    return strategies


@register_sharding(npu.npu_moe_token_permute_grad_v2.default)
def npu_moe_token_permute_grad_v2_strategy(grad_permuted_tokens, sorted_indices, tokens_size_0, tokens_dtype, num_topK, padded_mode=False):
    # func: npu_moe_token_permute_grad_v2(Tensor grad_permuted_tokens, Tensor sorted_indices,
    #                                      int tokens_size_0, ScalarType tokens_dtype, int num_topK, bool padded_mode=False) -> Tensor
    strategies = []

    # all replicate strategy
    replicate_strategy = (
        [Replicate()], # output
        [Replicate(), Replicate(), None, None, None, None] # input
    )
    strategies.append(replicate_strategy)

    # hidden_size dim sharding strategy
    hidden_size_sharding_strategy = (
        [Shard(1)],
        [Shard(1), Replicate(), None, None, None, None]
    )
    strategies.append(hidden_size_sharding_strategy)

    return strategies


@register_sharding(npu.npu_moe_token_unpermute.default)
def npu_moe_token_unpermute_strategy(permuted_tokens, sorted_indices, probs=None, padded_mode=False,
                                     restore_shape=None):
    # func: npu_moe_token_unpermute(Tensor permuted_tokens, Tensor sorted_indices, Tensor? probs=None,
    #                               bool padded_mode=False, int[]? restore_shape=None) -> Tensor
    strategies = []

    # all replicate strategy
    replicate_strategy = (
        [Replicate()], # output
        [Replicate(), Replicate(), None if probs is None else Replicate(), None, None] # input
    )
    strategies.append(replicate_strategy)

    # hidden_size dim sharding strategy
    hidden_size_sharding_strategy = (
        [Shard(1)],
        [Shard(1), Replicate(), None if probs is None else Replicate(), None, None]
    )
    strategies.append(hidden_size_sharding_strategy)

    return strategies


@register_sharding(npu._npu_moe_token_unpermute.default)
def _npu_moe_token_unpermute_strategy(permuted_tokens, sorted_indices, probs=None, padded_mode=False,
                                      restore_shape=None):
    # func: _npu_moe_token_unpermute(Tensor permuted_tokens, Tensor sorted_indices, Tensor? probs=None,
    #                                bool padded_mode=False, int[]? restore_shape=None)
    #                                -> (Tensor unpermuted_tokens, Tensor permuted_tokens_for_backward)
    strategies = []

    # all replicate strategy
    replicate_strategy = (
        [Replicate(), Replicate()], # output, saved permuted_tokens
        [Replicate(), Replicate(), None if probs is None else Replicate(), None, None] # input
    )
    strategies.append(replicate_strategy)

    # hidden_size dim sharding strategy
    hidden_size_sharding_strategy = (
        [Shard(1), Shard(1)],
        [Shard(1), Replicate(), None if probs is None else Replicate(), None, None]
    )
    strategies.append(hidden_size_sharding_strategy)

    return strategies


@register_sharding(npu.npu_moe_token_unpermute_grad.default)
def npu_moe_token_unpermute_grad_strategy(permuted_tokens, grad_unpermuted_tokens, sorted_indices, probs=None,
                                          padded_mode=False, restore_shape=None):
    # func: npu_moe_token_unpermute_grad(Tensor permuted_tokens, Tensor grad_unpermuted_tokens, Tensor sorted_indices,
    #                                    Tensor? probs=None, bool padded_mode=False, int[]? restore_shape=None)
    #                                   -> (Tensor, Tensor)
    strategies = []

    # all replicate strategy
    replicate_strategy = (
        [Replicate(), None if probs is None else Replicate()], # permuted_tokens grad, probs grad
        [Replicate(), Replicate(), Replicate(), None if probs is None else Replicate(), None, None] # input
    )
    strategies.append(replicate_strategy)

    # hidden_size dim sharding strategy
    hidden_size_sharding_strategy = (
        [Shard(1), Partial()],
        [Shard(1), Shard(1), Replicate(), None if probs is None else Replicate(), None, None]
    )
    strategies.append(hidden_size_sharding_strategy)

    return strategies


@register_sharding(npu.npu_moe_token_unpermute_grad_v2.default)
def npu_moe_token_unpermute_grad_v2_strategy(grad_unpermuted_tokens, sorted_indices, permuted_tokens_size_0,
                                             permuted_tokens_dtype, probs=None, padded_mode=False,
                                             restore_shape=None, permuted_tokens=None):
    # func: npu_moe_token_unpermute_grad_v2(Tensor grad_unpermuted_tokens, Tensor sorted_indices,
    #                                      int permuted_tokens_size_0, ScalarType permuted_tokens_dtype,
    #                                      Tensor? probs=None, bool padded_mode=False, int[]? restore_shape=None,
    #                                      Tensor? permuted_tokens=None) -> (Tensor, Tensor)
    strategies = []

    # all replicate strategy
    replicate_strategy = (
        [Replicate(), None if probs is None else Replicate()], # permuted_tokens grad, probs grad
        [Replicate(), Replicate(), None, None, None if probs is None else Replicate(), None, None,
         None if permuted_tokens is None else Replicate()] # input
    )
    strategies.append(replicate_strategy)

    # hidden_size dim sharding strategy
    hidden_size_sharding_strategy = (
        [Shard(1), Partial()],
        [Shard(1), Replicate(), None, None, None if probs is None else Replicate(), None, None,
         None if permuted_tokens is None else Shard(1)]
    )
    strategies.append(hidden_size_sharding_strategy)

    return strategies
