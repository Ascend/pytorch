import itertools
from typing import cast, Any, Dict, Tuple

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor.experimental import register_sharding
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._op_schema import (
    OpInfo,
    OpSchema,
    OutputSharding
)
from torch.distributed.tensor._redistribute import redistribute_local_tensor

import torch_npu

from ._common import (
    get_redistributed_local_args,
    get_redistributed_local_kwargs,
    get_empty_local_results
)

npu = torch.ops.npu


@register_sharding(npu.npu_fusion_attention.default)
# pylint:disable=huawei-too-many-arguments
def npu_fusion_attention_strategy(query, key, value, head_num, input_layout, pse=None, padding_mask=None,
                                  atten_mask=None, scale=1.0, keep_prob=1.0, pre_tockens=2147483647,
                                  next_tockens=2147483647, inner_precise=0, prefix=None, actual_seq_qlen=None,
                                  actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False,
                                  softmax_layout="", sink=None):
    # func: npu_fusion_attention(Tensor query, Tensor key, Tensor value, int head_num, str input_layout,
    #                            Tensor? pse=None, Tensor? padding_mask=None, Tensor? atten_mask=None, float scale=1.,
    #                            float keep_prob=1., int pre_tockens=2147483647, int next_tockens=2147483647,
    #                            int inner_precise=0, int[]? prefix=None, int[]? actual_seq_qlen=None,
    #                            int[]? actual_seq_kvlen=None, int sparse_mode=0, bool gen_mask_parallel=True,
    #                            bool sync=False, str softmax_layout="", Tensor? sink=None)
    #                           -> (Tensor, Tensor, Tensor, Tensor, int, int, int)
    strategies = []

    # all replicate strategy
    replicate_strategy = (
        [
            Replicate(),     # attention_out
            Replicate(),     # softmax_max
            Replicate(),     # softmax_sum
            Replicate(),     # softmax_out(reserve, unused now)
            None, None, None # others
        ],
        [
            Replicate(), # query
            Replicate(), # key
            Replicate(), # value
            None,        # head_num
            None,        # input_layout
            None if pse is None else Replicate(),          # pse
            None if padding_mask is None else Replicate(), # padding_mask
            None if atten_mask is None else Replicate(),   # atten_mask
            None, None, None, None, None, None, None, None, None, None, None, None, # others
            None if sink is None else Replicate() # sink
        ]
    )
    strategies.append(replicate_strategy)

    # only support sharding for sdpa currently, in which pse and padding_mask are not used
    # keep_prob < 1.0 may effect different results under sharding
    unused_args_in_sdpa = [pse, padding_mask, prefix, actual_seq_qlen, actual_seq_kvlen, sink]
    if not all(arg is None for arg in unused_args_in_sdpa) or keep_prob < 1.0:
        return strategies

    # input layout: BSH, SBH, BSND, BNSD, TND
    # atten_mask layout: BNSS, B1SS, 11SS, SS
    # dp sharding strategy
    if 'B' in input_layout:
        batch_dim = input_layout.index('B')
        atten_mask_sharding = None
        if atten_mask is not None:
            if atten_mask.ndim == 4 and atten_mask.shape[0] != 1: # BNSS, B1SS
                atten_mask_sharding = Shard(0)
            else: # 11SS, SS
                atten_mask_sharding = Replicate()
        dp_sharding_strategy = (
            [
                Shard(batch_dim), # attention_out
                Shard(0),         # softmax_max layout: BNS8
                Shard(0),         # softmax_sum layout: BNS8
                Replicate(),      # softmax_out(reserve, unused now)
                None, None, None  # others
            ],
            [
                Shard(batch_dim),    # query
                Shard(batch_dim),    # key
                Shard(batch_dim),    # value
                None,                # head_num
                None,                # input_layout
                None,                # pse
                None,                # padding_mask
                atten_mask_sharding, # atten_mask
                None, None, None, None, None, None, None, None, None, None, None, None, # others
                None # sink
            ]
        )
        strategies.append(dp_sharding_strategy)

    # add tp sharding strategy
    if 'N' in input_layout:
        head_dim = input_layout.index('N')
        atten_mask_sharding = None
        if atten_mask is not None:
            if atten_mask.ndim == 4 and atten_mask.shape[1] != 1: # BNSS
                atten_mask_sharding = Shard(1)
            else:
                atten_mask_sharding = Replicate() # B1SS, 11SS, SS
        tp_sharding_strategy = (
            [
                Shard(head_dim), # attention_out
                Shard(1),        # softmax_max layout: BNS8
                Shard(1),        # softmax_sum layout: BNS8
                Replicate(),     # softmax_out(reserve, unused now)
                None, None, None # others
            ],
            [
                Shard(head_dim),     # query
                Shard(head_dim),     # key
                Shard(head_dim),     # value
                None,                # head_num
                None,                # input_layout
                None,                # pse
                None,                # padding_mask
                atten_mask_sharding, # atten_mask
                None, None, None, None, None, None, None, None, None, None, None, None, # others
                None # sink
            ]
        )
        strategies.append(tp_sharding_strategy)

    return strategies


@register_sharding(npu.npu_fusion_attention_grad.default)
def npu_fusion_attention_grad_strategy(query, key, value, dy, head_num, input_layout, pse=None, padding_mask=None,
                                       atten_mask=None, softmax_max=None, softmax_sum=None, softmax_in=None,
                                       attention_in=None, scale_value=1., keep_prob=1., pre_tockens=2147483647,
                                       next_tockens=2147483647, inner_precise=0, seed=0, offset=0, numels=0,
                                       prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                                       gen_mask_parallel=True, sync=False, softmax_layout="", sink=None):
    # npu_fusion_attention_grad(Tensor query, Tensor key, Tensor value, Tensor dy, int head_num, str input_layout, *,
    #                           Tensor? pse=None, Tensor? padding_mask=None, Tensor? atten_mask=None,
    #                           Tensor? softmax_max=None, Tensor? softmax_sum=None, Tensor? softmax_in=None,
    #                           Tensor? attention_in=None, float scale_value=1., float keep_prob=1.,
    #                           int pre_tockens=2147483647, int next_tockens=2147483647, int inner_precise=0,
    #                           int seed=0, int offset=0, int numels=0, int[]? prefix=None,
    #                           int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None, int sparse_mode=0,
    #                           bool gen_mask_parallel=True, bool sync=False, str softmax_layout="", Tensor? sink=None)
    #                          -> (Tensor, Tensor, Tensor, Tensor, Tensor)
    strategies = []

    # all replicate strategy
    replicate_strategy = (
        [
            Replicate(), # grad_query
            Replicate(), # grad_key
            Replicate(), # grad_value
            Replicate(), # grad_pse(reserve, unused now)
            Replicate()  # grad_sink
        ],
        [
            Replicate(), # query
            Replicate(), # key
            Replicate(), # value
            Replicate(), # dy
            None,        # head_num
            None,        # input_layout
            None if pse is None else Replicate(),          # pse
            None if padding_mask is None else Replicate(), # padding_mask
            None if atten_mask is None else Replicate(),   # atten_mask
            None if softmax_max is None else Replicate(),  # softmax_max
            None if softmax_sum is None else Replicate(),  # softmax_sum
            None if softmax_in is None else Replicate(),   # softmax_in(reserve, unused now)
            None if attention_in is None else Replicate(), # attention_in
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, # others
            None if sink is None else Replicate() # sink
        ]
    )
    strategies.append(replicate_strategy)

    # only support sharding for sdpa currently, in which pse and padding_mask are not used
    # keep_prob < 1.0 may effect different results under sharding
    unused_args_in_sdpa = [pse, padding_mask, prefix, actual_seq_qlen, actual_seq_kvlen, sink]
    if not all(arg is None for arg in unused_args_in_sdpa) or keep_prob < 1.0:
        return strategies

    # input layout: BSH, SBH, BSND, BNSD, TND
    # atten_mask layout: BNSS, B1SS, 11SS, SS
    # dp sharding strategy
    if 'B' in input_layout:
        batch_dim = input_layout.index('B')
        atten_mask_sharding = None
        if atten_mask is not None:
            if atten_mask.ndim == 4 and atten_mask.shape[0] != 1: # BNSS, B1SS
                atten_mask_sharding = Shard(0)
            else: # 11SS, SS
                atten_mask_sharding = Replicate()
        dp_sharding_strategy = (
            [
                Shard(batch_dim), # grad_query
                Shard(batch_dim), # grad_key
                Shard(batch_dim), # grad_value
                Replicate(),      # grad_pse(reserve, unused now)
                Replicate()       # grad_sink(unsupported now)
            ],
            [
                Shard(batch_dim),    # query
                Shard(batch_dim),    # key
                Shard(batch_dim),    # value
                Shard(batch_dim),    # dy
                None,                # head_num
                None,                # input_layout
                None,                # pse
                None,                # padding_mask
                atten_mask_sharding, # atten_mask
                Shard(0) if softmax_max is not None else None,          # softmax_max layout: BNS8
                Shard(0) if softmax_sum is not None else None,          # softmax_sum layout: BNS8
                None if softmax_in is None else Replicate(),            # softmax_in(reserve, unused now)
                Shard(batch_dim) if attention_in is not None else None, # attention_in
                None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, # others
                None # sink
            ]
        )
        strategies.append(dp_sharding_strategy)

    # add tp sharding strategy
    if 'N' in input_layout:
        head_dim = input_layout.index('N')
        atten_mask_sharding = None
        if atten_mask is not None:
            if atten_mask.ndim == 4 and atten_mask.shape[1] != 1: # BNSS
                atten_mask_sharding = Shard(1)
            else:
                atten_mask_sharding = Replicate() # B1SS, 11SS, SS
        tp_sharding_strategy = (
            [
                Shard(head_dim), # grad_query
                Shard(head_dim), # grad_key
                Shard(head_dim), # grad_value
                Replicate(),     # grad_pse(reserve, unused now)
                Replicate()      # grad_sink(unsupported now)
            ],
            [
                Shard(head_dim),     # query
                Shard(head_dim),     # key
                Shard(head_dim),     # value
                Shard(head_dim),     # dy
                None,                # head_num
                None,                # input_layout
                None,                # pse
                None,                # padding_mask
                atten_mask_sharding, # atten_mask
                Shard(1) if softmax_max is not None else None,         # softmax_max layout: BNS8
                Shard(1) if softmax_sum is not None else None,         # softmax_sum layout: BNS8
                None if softmax_in is None else Replicate(),           # softmax_in(reserve, unused now)
                Shard(head_dim) if attention_in is not None else None, # attention_in
                None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, # others
                None # sink
            ]
        )
        strategies.append(tp_sharding_strategy)

    return strategies


@register_sharding(npu.npu_fusion_attention_v3.default)
# pylint:disable=huawei-too-many-arguments
def npu_fusion_attention_v3_strategy(query, key, value, head_num, input_layout, pse=None, padding_mask=None,
                                  atten_mask=None, scale=1.0, keep_prob=1.0, pre_tockens=2147483647,
                                  next_tockens=2147483647, inner_precise=0, prefix=None, actual_seq_qlen=None,
                                  actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False,
                                  softmax_layout="", sink=None):
    strategies = []

    # all replicate strategy
    replicate_strategy = (
        [
            Replicate(),     # attention_out
            Replicate(),     # softmax_max
            Replicate(),     # softmax_sum
            Replicate(),     # softmax_out(reserve, unused now)
            Replicate(),     # seed
            Replicate()      # offset
        ],
        [
            Replicate(), # query
            Replicate(), # key
            Replicate(), # value
            None,        # head_num
            None,        # input_layout
            None if pse is None else Replicate(),          # pse
            None if padding_mask is None else Replicate(), # padding_mask
            None if atten_mask is None else Replicate(),   # atten_mask
            None, None, None, None, None, None, # others
            None if actual_seq_qlen is None else Replicate(),    # actual_seq_qlen
            None if actual_seq_kvlen is None else Replicate(),   # actual_seq_kvlen
            None, None, None, None, # others
            None if sink is None else Replicate() # sink
        ]
    )
    strategies.append(replicate_strategy)

    # only support sharding for sdpa currently, in which pse and padding_mask are not used
    # keep_prob < 1.0 may effect different results under sharding
    unused_args_in_sdpa = [pse, padding_mask, prefix, actual_seq_qlen, actual_seq_kvlen, sink]
    if not all(arg is None for arg in unused_args_in_sdpa) or keep_prob < 1.0:
        return strategies

    # input layout: BSH, SBH, BSND, BNSD, TND
    # atten_mask layout: BNSS, B1SS, 11SS, SS
    # dp sharding strategy
    if 'B' in input_layout:
        batch_dim = input_layout.index('B')
        atten_mask_sharding = None
        if atten_mask is not None:
            if atten_mask.ndim == 4 and atten_mask.shape[0] != 1: # BNSS, B1SS
                atten_mask_sharding = Shard(0)
            else: # 11SS, SS
                atten_mask_sharding = Replicate()
        dp_sharding_strategy = (
            [
                Shard(batch_dim), # attention_out
                Shard(0),         # softmax_max layout: BNS8
                Shard(0),         # softmax_sum layout: BNS8
                Replicate(),      # softmax_out(reserve, unused now)
                Replicate(),      # seed
                Replicate()       # offset
            ],
            [
                Shard(batch_dim),    # query
                Shard(batch_dim),    # key
                Shard(batch_dim),    # value
                None,                # head_num
                None,                # input_layout
                None,                # pse
                None,                # padding_mask
                atten_mask_sharding, # atten_mask
                None, None, None, None, None, None, # others
                None if actual_seq_qlen is None else Replicate(),    # actual_seq_qlen
                None if actual_seq_kvlen is None else Replicate(),   # actual_seq_kvlen
                None, None, None, None, # others
                None # sink
            ]
        )
        strategies.append(dp_sharding_strategy)

    # add tp sharding strategy
    if 'N' in input_layout:
        head_dim = input_layout.index('N')
        atten_mask_sharding = None
        if atten_mask is not None:
            if atten_mask.ndim == 4 and atten_mask.shape[1] != 1: # BNSS
                atten_mask_sharding = Shard(1)
            else:
                atten_mask_sharding = Replicate() # B1SS, 11SS, SS
        tp_sharding_strategy = (
            [
                Shard(head_dim), # attention_out
                Shard(1),        # softmax_max layout: BNS8
                Shard(1),        # softmax_sum layout: BNS8
                Replicate(),     # softmax_out(reserve, unused now)
                Replicate(),     # seed
                Replicate()      # offset
            ],
            [
                Shard(head_dim),     # query
                Shard(head_dim),     # key
                Shard(head_dim),     # value
                None,                # head_num
                None,                # input_layout
                None,                # pse
                None,                # padding_mask
                atten_mask_sharding, # atten_mask
                None, None, None, None, None, None, # others
                None if actual_seq_qlen is None else Replicate(),    # actual_seq_qlen
                None if actual_seq_kvlen is None else Replicate(),   # actual_seq_kvlen
                None, None, None, None, # others
                None # sink
            ]
        )
        strategies.append(tp_sharding_strategy)

    return strategies


@register_sharding(npu.npu_fusion_attention_grad_v3.default)
def npu_fusion_attention_grad_v3_strategy(query, key, value, dy, head_num, input_layout, pse=None, padding_mask=None,
                                       atten_mask=None, softmax_max=None, softmax_sum=None, softmax_in=None,
                                       attention_in=None, scale_value=1., keep_prob=1., pre_tockens=2147483647,
                                       next_tockens=2147483647, inner_precise=0, seed=None, offset=None,
                                       prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                                       gen_mask_parallel=True, sync=False, softmax_layout="", sink=None):
    strategies = []

    # all replicate strategy
    replicate_strategy = (
        [
            Replicate(), # grad_query
            Replicate(), # grad_key
            Replicate(), # grad_value
            Replicate(), # grad_pse(reserve, unused now)
            Replicate()  # grad_sink
        ],
        [
            Replicate(), # query
            Replicate(), # key
            Replicate(), # value
            Replicate(), # dy
            None,        # head_num
            None,        # input_layout
            None if pse is None else Replicate(),          # pse
            None if padding_mask is None else Replicate(), # padding_mask
            None if atten_mask is None else Replicate(),   # atten_mask
            None if softmax_max is None else Replicate(),  # softmax_max
            None if softmax_sum is None else Replicate(),  # softmax_sum
            None if softmax_in is None else Replicate(),   # softmax_in(reserve, unused now)
            None if attention_in is None else Replicate(), # attention_in
            None, None, None, None, None, # others
            None if seed is None else Replicate(),        # seed
            None if offset is None else Replicate(),      # offset
            None, 
            None if actual_seq_qlen is None else Replicate(),  # actual_seq_qlen
            None if actual_seq_kvlen is None else Replicate(), # actual_seq_kvlen
            None, None, None, None, # others
            None if sink is None else Replicate() # sink
        ]
    )
    strategies.append(replicate_strategy)

    # only support sharding for sdpa currently, in which pse and padding_mask are not used
    # keep_prob < 1.0 may effect different results under sharding
    unused_args_in_sdpa = [pse, padding_mask, prefix, actual_seq_qlen, actual_seq_kvlen, sink]
    if not all(arg is None for arg in unused_args_in_sdpa) or keep_prob < 1.0:
        return strategies

    # input layout: BSH, SBH, BSND, BNSD, TND
    # atten_mask layout: BNSS, B1SS, 11SS, SS
    # dp sharding strategy
    if 'B' in input_layout:
        batch_dim = input_layout.index('B')
        atten_mask_sharding = None
        if atten_mask is not None:
            if atten_mask.ndim == 4 and atten_mask.shape[0] != 1: # BNSS, B1SS
                atten_mask_sharding = Shard(0)
            else: # 11SS, SS
                atten_mask_sharding = Replicate()
        dp_sharding_strategy = (
            [
                Shard(batch_dim), # grad_query
                Shard(batch_dim), # grad_key
                Shard(batch_dim), # grad_value
                Replicate(),      # grad_pse(reserve, unused now)
                Replicate()       # grad_sink(unsupported now)
            ],
            [
                Shard(batch_dim),    # query
                Shard(batch_dim),    # key
                Shard(batch_dim),    # value
                Shard(batch_dim),    # dy
                None,                # head_num
                None,                # input_layout
                None,                # pse
                None,                # padding_mask
                atten_mask_sharding, # atten_mask
                Shard(0) if softmax_max is not None else None,          # softmax_max layout: BNS8
                Shard(0) if softmax_sum is not None else None,          # softmax_sum layout: BNS8
                None if softmax_in is None else Replicate(),            # softmax_in(reserve, unused now)
                Shard(batch_dim) if attention_in is not None else None, # attention_in
                None, None, None, None, None, # others
                None if seed is None else Replicate(),        # seed
                None if offset is None else Replicate(),      # offset
                None, 
                None if actual_seq_qlen is None else Replicate(),  # actual_seq_qlen
                None if actual_seq_kvlen is None else Replicate(), # actual_seq_kvlen
                None, None, None, None, # others
                None # sink
            ]
        )
        strategies.append(dp_sharding_strategy)

    # add tp sharding strategy
    if 'N' in input_layout:
        head_dim = input_layout.index('N')
        atten_mask_sharding = None
        if atten_mask is not None:
            if atten_mask.ndim == 4 and atten_mask.shape[1] != 1: # BNSS
                atten_mask_sharding = Shard(1)
            else:
                atten_mask_sharding = Replicate() # B1SS, 11SS, SS
        tp_sharding_strategy = (
            [
                Shard(head_dim), # grad_query
                Shard(head_dim), # grad_key
                Shard(head_dim), # grad_value
                Replicate(),     # grad_pse(reserve, unused now)
                Replicate()      # grad_sink(unsupported now)
            ],
            [
                Shard(head_dim),     # query
                Shard(head_dim),     # key
                Shard(head_dim),     # value
                Shard(head_dim),     # dy
                None,                # head_num
                None,                # input_layout
                None,                # pse
                None,                # padding_mask
                atten_mask_sharding, # atten_mask
                Shard(1) if softmax_max is not None else None,         # softmax_max layout: BNS8
                Shard(1) if softmax_sum is not None else None,         # softmax_sum layout: BNS8
                None if softmax_in is None else Replicate(),           # softmax_in(reserve, unused now)
                Shard(head_dim) if attention_in is not None else None, # attention_in
                None, None, None, None, None, # others
                None if seed is None else Replicate(),        # seed
                None if offset is None else Replicate(),      # offset
                None, 
                None if actual_seq_qlen is None else Replicate(),  # actual_seq_qlen
                None if actual_seq_kvlen is None else Replicate(), # actual_seq_kvlen
                None, None, None, None, # others
                None # sink
            ]
        )
        strategies.append(tp_sharding_strategy)

    return strategies


def _infer_npu_fusion_attention_grad_kwargs_spec(
    op_schema: OpSchema,
    output_sharding: OutputSharding
) -> Dict[str, DTensorSpec]:
    input_layout = op_schema.args_schema[5]
    batch_dim = input_layout.index('B') if 'B' in input_layout else None
    dp_shard = Shard(batch_dim) if batch_dim is not None else None
    head_dim = input_layout.index('N') if 'N' in input_layout else None
    tp_shard = Shard(head_dim) if head_dim is not None else None
    output_spec = output_sharding.output_spec[0]
    kwargs_spec = {}
    for key, spec in op_schema.kwargs_schema.items():
        if not isinstance(spec, DTensorSpec):
            kwargs_spec[key] = spec
            continue

        target_placement = []
        for placement in output_spec.placements:
            if placement == Replicate():
                target_placement.append(Replicate())
            elif placement == dp_shard:
                if key == 'atten_mask':
                    atten_mask = op_schema.kwargs_schema[key]
                    if atten_mask.ndim == 4 and atten_mask.shape[0] != 1: # BNSS, B1SS
                        target_placement.append(dp_shard)
                    else: # 11SS, SS
                        target_placement.append(Replicate())
                elif key == 'softmax_max' or key == 'softmax_sum':
                    target_placement.append(Shard(0))
                elif key == 'attention_in':
                    target_placement.append(dp_shard)
                else: # softmax_in
                    target_placement.append(Replicate())
            elif placement == tp_shard:
                if key == 'atten_mask':
                    atten_mask = op_schema.kwargs_schema[key]
                    if atten_mask.ndim == 4 and atten_mask.shape[1] != 1: # BNSS
                        target_placement.append(tp_shard)
                    else: # B1SS, 11SS, SS
                        target_placement.append(Replicate())
                elif key == 'softmax_max' or key == 'softmax_sum':
                    target_placement.append(Shard(1))
                elif key == 'attention_in':
                    target_placement.append(tp_shard)
                else: # softmax_in
                    target_placement.append(Replicate())
            else:
                raise ValueError(
                    f"Unexpected placement {placement} for npu_fusion_attention_grad in layout {input_layout}."
                )

        kwargs_spec[key] = DTensorSpec(
            mesh=spec.mesh,
            placements=target_placement,
            tensor_meta=spec.tensor_meta
        )

    return kwargs_spec


def _npu_fusion_attention_handler(
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
) -> object:
    def npu_attention_input_fn(
            mesh: DeviceMesh, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        all_args = []

        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, torch.Tensor) and not isinstance(arg, DTensor):
                arg = DTensor.from_local(arg, mesh, [Replicate()], run_check=False)

            all_args.append(arg)

        new_args = tuple(all_args[0: len(args)])
        new_kwargs = dict(zip(kwargs.keys(), all_args[len(args):]))

        return new_args, new_kwargs

    runtime_schema_info = (
        DTensor._op_dispatcher.sharding_propagator.op_to_schema_info.get(op_call, None)
    )

    if runtime_schema_info is not None and runtime_schema_info.needs_pytree:
        try:
            from torch.utils import _cxx_pytree as pytree
        except ImportError:
            from torch.utils import _pytree as pytree  # type: ignore[no-redef]
        from typing import Sequence

        tree_args, args_spec = pytree.tree_flatten(args)
        args_list: Sequence[object] = tree_args
    else:
        args_list, args_spec = args, None

    args, kwargs = npu_attention_input_fn(args_list[0].device_mesh, *args, **kwargs)

    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding

    mesh = op_info.mesh
    participating = mesh.get_coordinate() is not None
    if participating:
        # computation that happens in the current rank of the mesh, normal case
        local_args = get_redistributed_local_args(op_info, output_sharding)
        local_kwargs = op_info.local_kwargs
        if op_call == npu.npu_fusion_attention.default or op_call == npu.npu_fusion_attention_v3.default:
            # if sharding head_dim in qkv, need recalculate head_num in local args
            input_layout = op_info.local_args[4]
            if 'N' in input_layout:
                head_dim = input_layout.index('N')
                local_args = list(local_args)
                local_query = local_args[0]
                local_args[3] = local_query.size(head_dim)
                local_args = tuple(local_args)

            # run local op computation with potentially modified args/kwargs
            if op_call == npu.npu_fusion_attention.default:
                local_results = torch_npu.npu_fusion_attention(
                    *local_args, **local_kwargs
                )
            else:
                local_results = torch_npu.npu_fusion_attention_v3(
                    *local_args, **local_kwargs
                )
        elif op_call == npu.npu_fusion_attention_grad.default or op_call == npu.npu_fusion_attention_grad_v3.default:
            local_kwargs = get_redistributed_local_kwargs(
                _infer_npu_fusion_attention_grad_kwargs_spec, op_info, output_sharding
            )
            # if sharding head_dim in qkv, need recalculate head_num in local args
            input_layout = op_info.local_args[5]
            if 'N' in input_layout:
                head_dim = input_layout.index('N')
                local_args = list(local_args)
                local_query = local_args[0]
                local_args[4] = local_query.size(head_dim)
                local_args = tuple(local_args)
            if op_call == npu.npu_fusion_attention_grad.default:
                local_results = torch_npu.npu_fusion_attention_grad(
                    *local_args, **local_kwargs
                )
            else:
                local_results = torch_npu.npu_fusion_attention_grad_v3(
                    *local_args, **local_kwargs
                )
        else:
            raise NotImplementedError(
                "_npu_fusion_attention_handler only supports npu_fusion_attention and npu_fusion_attention_grad now."
            )
    else:
        # For a non-participating device (happens on rank that does not belong to the device mesh),
        # return empty tensor(s) with correct dtype.
        spec = output_sharding.output_spec

        def default_tensor(spec: DTensorSpec) -> torch.Tensor:
            if spec.tensor_meta is not None:
                shape = spec.tensor_meta.shape
                dtype = spec.tensor_meta.dtype
                if len(shape) == 0:
                    # scalar tensor
                    return torch.zeros((), dtype=dtype)
                else:
                    # non-scalar tensor
                    return torch.tensor([], dtype=dtype)
            else:
                raise RuntimeError(f"{spec} has no tensor metadata.")

        # only have Tensor and int outputs here
        local_results = [default_tensor(s) if s is not None else 0 for s in spec]

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


customized_ops = {
    npu.npu_fusion_attention.default: _npu_fusion_attention_handler,
    npu.npu_fusion_attention_grad.default: _npu_fusion_attention_handler,
    npu.npu_fusion_attention_v3.default: _npu_fusion_attention_handler,
    npu.npu_fusion_attention_grad_v3.default: _npu_fusion_attention_handler,
}

old_handlers = DTensor._op_dispatcher._custom_op_handlers
DTensor._op_dispatcher._custom_op_handlers = {**old_handlers, **customized_ops}
