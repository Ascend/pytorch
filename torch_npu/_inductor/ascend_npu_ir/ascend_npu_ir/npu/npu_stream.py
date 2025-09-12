import torch
import torch_npu
import torch.library
from torch.library import Library

from typing import Callable, Optional, Sequence, List, Tuple

NPU_STREAMS = {}
NPU_EVENTS = {}

# create a library to hold the custom op
npu_stream_lib = Library("npu_stream", "FRAGMENT")  # noqa

def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str],
    fake_impl: Optional[Callable] = None,
    target_lib: Optional[Library] = None,
    dispatch_key: str = "PrivateUse1",
    tags: tuple[torch.Tag, ...] = (),
):
    """
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    for more details.

    By default, the custom op is registered to the vLLM library. If you
    want to register it to a different library, you can pass the library
    object to the `target_lib` argument.

    IMPORTANT: the lifetime of the operator is tied to the lifetime of the
    library object. If you want to bind the operator to a different library,
    make sure the library object is alive when the operator is used.
    """
    import torch.library
    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(op_func,
                                                mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        import torch._custom_op.impl
        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)
    my_lib = target_lib or npu_stream_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)

class StreamResgistrator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def register_npu_stream(stream: torch.npu.Stream, tag: str = '0'):
        NPU_STREAMS[tag] = stream

    @staticmethod
    def register_npu_event(event: torch.npu.Event, tag: str = '0'):
        NPU_EVENTS[tag] = event

def npu_set_stream(
    dependency: Sequence[torch.Tensor], 
    stream_tag: str,
    ) -> List[torch.Tensor]:
    stream = NPU_STREAMS[stream_tag]
    torch_npu.npu.utils.set_stream(stream)
    return dependency

def npu_set_stream_fake(
    dependency: Sequence[torch.Tensor], 
    stream_tag: str,
    ) -> List[torch.Tensor]:
    return dependency

direct_register_custom_op(
    op_name="npu_set_stream",
    op_func=npu_set_stream,
    mutates_args=[],
    fake_impl=npu_set_stream_fake,
    dispatch_key='PrivateUse1'
)

def npu_event_record(
    dependency: Sequence[torch.Tensor], 
    event_tag: str,
    stream_tag: str
    ) -> List[torch.Tensor]:
    event = NPU_EVENTS[event_tag]
    stream = NPU_STREAMS[stream_tag]
    event.record(stream)
    return dependency

def npu_event_record_fake(
    dependency: Sequence[torch.Tensor], 
    event_tag: str,
    stream_tag: str
    ) -> List[torch.Tensor]:
    return dependency

direct_register_custom_op(
    op_name="npu_event_record",
    op_func=npu_event_record,
    mutates_args=[],
    fake_impl=npu_event_record_fake,
    dispatch_key='PrivateUse1'
)

def npu_event_wait(
    dependency: Sequence[torch.Tensor], 
    event_tag: str,
    ) -> List[torch.Tensor]:
    event = NPU_EVENTS[event_tag]
    event.wait()
    return dependency

def npu_event_wait_fake(
    dependency: Sequence[torch.Tensor], 
    event_tag: str,
    ) -> List[torch.Tensor]:
    return dependency

direct_register_custom_op(
    op_name="npu_event_wait",
    op_func=npu_event_wait,
    mutates_args=[],
    fake_impl=npu_event_wait_fake,
    dispatch_key='PrivateUse1'
)

def graph_break(
    dependency: Sequence[torch.Tensor], 
    ) -> List[torch.Tensor]:
    return dependency

def graph_break_fake(
    dependency: Sequence[torch.Tensor], 
    ) -> List[torch.Tensor]:
    return dependency

utils_lib = Library("npu_utils", "FRAGMENT")  # noqa

direct_register_custom_op(
    op_name="graph_break",
    op_func=graph_break,
    mutates_args=[],
    target_lib=utils_lib,
    fake_impl=graph_break_fake,
    dispatch_key='PrivateUse1'
)

def npu_wait_stream(
    dependency: Sequence[torch.Tensor], 
    stream1_tag: str,
    stream2_tag: str,
    ) -> List[torch.Tensor]:
    stream1 = NPU_STREAMS[stream1_tag]
    stream2 = NPU_STREAMS[stream2_tag]
    stream1.wait_stream(stream2)
    return dependency

def npu_wait_stream_fake(
    dependency: Sequence[torch.Tensor], 
    stream1_tag: str,
    stream2_tag: str,
    ) -> List[torch.Tensor]:
    return dependency

direct_register_custom_op(
    op_name="npu_wait_stream",
    op_func=npu_wait_stream,
    mutates_args=[],
    fake_impl=npu_wait_stream_fake,
    dispatch_key='PrivateUse1'
)


def graph_break(
    *args
):
    outputs = []
    for inp in args:
        outputs.append(torch.ops.npu_utils.graph_break(inp))
    return outputs

inductor_npu_lib = Library("inductor_npu", "FRAGMENT")  # noqa

def npu_fusion_attention(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    head_num: int, 
    input_layout: str, 
    pse: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    scale: float = 1.0, 
    keep_prob: float = 1.0, 
    pre_tockens: int = 2147483647, 
    next_tockens: int = 2147483647,
    inner_precise: int = 0, 
    prefix: Optional[torch.Tensor] = None, 
    actual_seq_qlen: Optional[torch.Tensor] = None, 
    actual_seq_kvlen: Optional[torch.Tensor] = None, 
    sparse_mode: int = 0,
    gen_mask_parallel: bool = True, 
    sync: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix = prefix.tolist() if prefix is not None else prefix
    actual_seq_qlen = actual_seq_qlen.tolist() if actual_seq_qlen is not None else actual_seq_qlen
    actual_seq_kvlen = actual_seq_kvlen.tolist() if actual_seq_kvlen is not None else actual_seq_kvlen
    attention_score, softmax_max, softmax_sum, softmax_out, seed, offset, numels = torch.ops.npu.npu_fusion_attention(
        query, 
        key, 
        value, 
        head_num, 
        input_layout, 
        pse=pse,
        padding_mask=padding_mask,
        atten_mask=atten_mask,
        scale=scale, 
        keep_prob=keep_prob, 
        pre_tockens=pre_tockens, 
        next_tockens=next_tockens,
        inner_precise=inner_precise, 
        prefix=prefix, 
        actual_seq_qlen=actual_seq_qlen, 
        actual_seq_kvlen=actual_seq_kvlen, 
        sparse_mode=sparse_mode,
        gen_mask_parallel=gen_mask_parallel, 
        sync=sync
    )

    seed = torch.tensor([seed], device='npu', dtype=torch.int64)
    offset = torch.tensor([offset], device='npu', dtype=torch.int64)
    numels = torch.tensor([numels], device='npu', dtype=torch.int64)

    return attention_score, softmax_max, softmax_sum, softmax_out, seed, offset, numels

def npu_fusion_attention_fake(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    head_num: int, 
    input_layout: str, 
    pse: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    scale: float = 1.0, 
    keep_prob: float = 1.0, 
    pre_tockens: int = 2147483647, 
    next_tockens: int = 2147483647,
    inner_precise: int = 0, 
    prefix: Optional[torch.Tensor] = None, 
    actual_seq_qlen: Optional[torch.Tensor] = None, 
    actual_seq_kvlen: Optional[torch.Tensor] = None, 
    sparse_mode: int = 0,
    gen_mask_parallel: bool = True, 
    sync: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = query.size(0)
    N = head_num
    S1 = query.size(2)
    S2 = key.size(2)

    if input_layout == "BSH":
        B = query.size(0)
        S1 = query.size(1)
        S2 = key.size(1)

    if input_layout == "SBH":
        B = query.size(1)
        S1 = query.size(0)
        S2 = key.size(0)

    attention_score = torch.empty_like(query, dtype=query.dtype, device=query.device).contiguous()
    softmax_max = torch.empty([B, head_num, S1, 8], dtype=torch.float32, device=query.device)
    softmax_sum = torch.empty([B, head_num, S1, 8], dtype=torch.float32, device=query.device)
    softmax_out = torch.empty([0], dtype=query.dtype, device=query.device)
    seed = torch.empty([1], dtype=torch.int64, device=query.device)
    offset = torch.empty([1], dtype=torch.int64, device=query.device)
    numels = torch.empty([1], dtype=torch.int64, device=query.device)

    return (attention_score,
            softmax_max,
            softmax_sum,
            softmax_out,
            seed,
            offset,
            numels)

direct_register_custom_op(
    op_name="npu_fusion_attention",
    op_func=npu_fusion_attention,
    mutates_args=[],
    target_lib=inductor_npu_lib,
    fake_impl=npu_fusion_attention_fake,
    dispatch_key='PrivateUse1'
)


def npu_fusion_attention_grad(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    dy: torch.Tensor, 
    head_num: int, 
    input_layout: str, 
    *, 
    pse: Optional[torch.Tensor] = None, 
    padding_mask: Optional[torch.Tensor] = None, 
    atten_mask: Optional[torch.Tensor] = None,
    softmax_max: Optional[torch.Tensor] = None, 
    softmax_sum: Optional[torch.Tensor] = None, 
    softmax_in: Optional[torch.Tensor] = None, 
    attention_in: Optional[torch.Tensor] = None, 
    scale_value: float = 1.0,
    keep_prob: float = 1.0, 
    pre_tockens: int = 2147483647, 
    next_tockens: int = 2147483647, 
    inner_precise: int = 0, 
    seed: Optional[torch.Tensor] = None, 
    offset: Optional[torch.Tensor] = None,
    numels: Optional[torch.Tensor] = None, 
    prefix: Optional[torch.Tensor] = None, 
    actual_seq_qlen: Optional[torch.Tensor] = None,
    actual_seq_kvlen: Optional[torch.Tensor] = None, 
    sparse_mode: int = 0,
    gen_mask_parallel: bool = True, 
    sync: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix = prefix.tolist() if prefix is not None else prefix
    actual_seq_qlen = actual_seq_qlen.tolist() if actual_seq_qlen is not None else actual_seq_qlen
    actual_seq_kvlen = actual_seq_kvlen.tolist() if actual_seq_kvlen is not None else actual_seq_kvlen

    seed = seed.item()
    offset = offset.item()
    numels = numels.item()

    dq, dk, dv, dpse = torch.ops.npu.npu_fusion_attention_grad(
        query, key, value, dy, head_num, input_layout, pse=pse, padding_mask=padding_mask, atten_mask=atten_mask,
        softmax_max=softmax_max, softmax_sum=softmax_sum, softmax_in=softmax_in, attention_in=attention_in, scale_value=scale_value,
        keep_prob=keep_prob, pre_tockens=pre_tockens, next_tockens=next_tockens, inner_precise=inner_precise, seed=seed, offset=offset,
        numels=numels, prefix=prefix, actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen, sparse_mode=sparse_mode,
        gen_mask_parallel=gen_mask_parallel, sync=sync
    )

    return dq, dk, dv, dpse

def npu_fusion_attention_grad_fake(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    dy: torch.Tensor, 
    head_num: int, 
    input_layout: str, 
    *, 
    pse: Optional[torch.Tensor] = None, 
    padding_mask: Optional[torch.Tensor] = None, 
    atten_mask: Optional[torch.Tensor] = None,
    softmax_max: Optional[torch.Tensor] = None, 
    softmax_sum: Optional[torch.Tensor] = None, 
    softmax_in: Optional[torch.Tensor] = None, 
    attention_in: Optional[torch.Tensor] = None, 
    scale_value: float = 1.0,
    keep_prob: float = 1.0, 
    pre_tockens: int = 2147483647, 
    next_tockens: int = 2147483647, 
    inner_precise: int = 0, 
    seed: Optional[torch.Tensor] = None, 
    offset: Optional[torch.Tensor] = None,
    numels: Optional[torch.Tensor] = None, 
    prefix: Optional[torch.Tensor] = None,
    actual_seq_qlen: Optional[torch.Tensor] = None,
    actual_seq_kvlen: Optional[torch.Tensor] = None, 
    sparse_mode: int = 0,
    gen_mask_parallel: bool = True, 
    sync: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty_like(query, dtype=query.dtype, device=query.device).contiguous()
    dk = torch.empty_like(key, dtype=query.dtype, device=query.device).contiguous()
    dv = torch.empty_like(value, dtype=query.dtype, device=query.device).contiguous()
    dpse = torch.empty([0], dtype=query.dtype, device=query.device).contiguous()
    return dq, dk, dv, dpse if pse else None

direct_register_custom_op(
    op_name="npu_fusion_attention_grad",
    op_func=npu_fusion_attention_grad,
    mutates_args=[],
    target_lib=inductor_npu_lib,
    fake_impl=npu_fusion_attention_grad_fake,
    dispatch_key='PrivateUse1'
)

class InductorNpuAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, head_num, input_layout, pse=None, padding_mask=None, atten_mask=None, scale=1.0,
                keep_prob=1.0, pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, prefix=None,
                actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False):
        attention_score, softmax_max, softmax_sum, softmax_out, seed, offset, numels = torch.ops.inductor_npu.npu_fusion_attention(
            query, key, value, head_num, input_layout, pse=pse, padding_mask=padding_mask, atten_mask=atten_mask,
            scale=scale, keep_prob=keep_prob, pre_tockens=pre_tockens, next_tockens=next_tockens,
            inner_precise=inner_precise, prefix=prefix, actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen, sparse_mode=sparse_mode, gen_mask_parallel=gen_mask_parallel, sync=sync
        )
        ctx.save_for_backward(query, key, value, pse, padding_mask, atten_mask, actual_seq_qlen, actual_seq_kvlen,\
                              softmax_max, softmax_sum, softmax_out, attention_score, seed, offset, numels)
        ctx.head_num = head_num
        ctx.input_layout = input_layout
        ctx.scale = scale
        ctx.keep_prob = keep_prob
        ctx.pre_tockens = pre_tockens
        ctx.next_tockens = next_tockens
        ctx.inner_precise = inner_precise
        ctx.prefix = prefix
        # ctx.actual_seq_qlen = actual_seq_qlen
        # ctx.actual_seq_kvlen = actual_seq_kvlen
        ctx.sparse_mode = sparse_mode
        ctx.gen_mask_parallel = gen_mask_parallel
        ctx.sync = sync

        return attention_score, softmax_max, softmax_sum, softmax_out, seed, offset, numels

    @staticmethod
    def backward(ctx, grad_attention_score, grad_softmax_max, grad_softmax_sum, grad_softmax_out, grad_seed, grad_offset, grad_numels):
        query, key, value, pse, padding_mask, atten_mask, actual_seq_qlen, actual_seq_kvlen, \
            softmax_max, softmax_sum, softmax_out, attention_score, seed, offset, numels = ctx.saved_tensors
        grad_query, grad_key, grad_value, grad_pse = torch.ops.inductor_npu.npu_fusion_attention_grad(
            query, key, value, grad_attention_score, ctx.head_num, ctx.input_layout, pse=pse, padding_mask=padding_mask,
            atten_mask=atten_mask, softmax_max=softmax_max, softmax_sum=softmax_sum, softmax_in=softmax_out, attention_in=attention_score,
            scale_value=ctx.scale, keep_prob=ctx.keep_prob, pre_tockens=ctx.pre_tockens, next_tockens=ctx.next_tockens,
            inner_precise=ctx.inner_precise, seed=seed, offset=offset, numels=numels, prefix=None,
            actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen, sparse_mode=ctx.sparse_mode,
            gen_mask_parallel=ctx.gen_mask_parallel, sync=ctx.sync
        )
        return (
        grad_query, grad_key, grad_value, None, None, grad_pse, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None)
    
def inductor_npu_fusion_attention(query, key, value, head_num, input_layout, pse=None, padding_mask=None,
                               atten_mask=None, scale=1.0, keep_prob=1.0, pre_tockens=2147483647,
                               next_tockens=2147483647,
                               inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                               gen_mask_parallel=True, sync=False):
    return InductorNpuAttentionFunction.apply(query, key, value, head_num, input_layout, pse, padding_mask,
                                           atten_mask, scale, keep_prob, pre_tockens, next_tockens,
                                           inner_precise, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode,
                                           gen_mask_parallel, sync)

def apply_inductor_npu_attention_patch():
    torch.ops.npu.npu_fusion_attention = inductor_npu_fusion_attention
