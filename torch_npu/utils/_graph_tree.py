import functools
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import torch
from torch.utils._ordered_set import OrderedSet
from torch._dynamo import utils as dynamo_utils
from torch._inductor import config
from torch._inductor.compile_fx import (
    get_input_idxs_to_check,
    index_expanded_dims_and_copy_,
    static_input,
)
from torch._inductor.cudagraph_utils import (
    _get_use_stack_trace,
    format_default_skip_message,
    PlaceholderInfo,
)
from torch._inductor.output_code import get_expanded_dims
from torch._inductor.utils import (
    align_inputs_from_check_idxs,
    copy_misaligned_inputs,
    remove_unaligned_input_idxs,
    InputType,
)


def npugraph_mark_step_begin():
    from torch_npu.npu._graph_tree import mark_step_begin
    mark_step_begin()


def check_multiple_devices_or_any_cpu_nodes(
    device_node_mapping: Dict[torch.device, torch.fx.Node]
) -> Optional[str]:
    cpu_node = device_node_mapping.get(torch.device("cpu"))
    if cpu_node:
        msg = f"cpu device ({cpu_node.name})"
        stack_trace = _get_use_stack_trace(cpu_node)
        if stack_trace:
            return format_default_skip_message(f"{msg}. Found from : \n {stack_trace}")
        return format_default_skip_message(msg)

    if (
        len(device_node_mapping) == 1
        and next(iter(device_node_mapping.keys())).type == "npu"
    ):
        return None

    keys_repr = (repr(key) for key in device_node_mapping.keys())
    return format_default_skip_message(f"multiple devices: {', '.join(keys_repr)}")


def npugraphify(
    model: Callable[..., Any],
    static_input_idxs: Sequence[int] = (),
    *,
    device_index: int,
    stack_traces: List[Optional[str]],
    is_backward: bool,
    is_inference: bool,
    constants: Tuple[torch.Tensor, ...] = (),
    placeholders: Sequence[PlaceholderInfo] = (),
    mutated_input_idxs: Tuple[int, ...] = (),
) -> Callable[..., Any]:
    from torch_npu.npu._graph_tree import npugraphify_impl as new_npugraphify_impl
    npugraphify_fn: Callable[..., Any]
    if config.triton.cudagraph_trees:
        npugraphify_fn = functools.partial(
            new_npugraphify_impl,
            device_index=device_index,
            stack_traces=stack_traces,
            is_backward=is_backward,
            is_inference=is_inference,
            constants=constants,
            placeholders=placeholders,
            mutated_input_idxs=mutated_input_idxs,
        )
    else:
        npugraphify_fn = npugraphify_impl

    compiled_fn = None

    def run(new_inputs: Sequence[InputType]) -> Any:
        nonlocal compiled_fn
        if compiled_fn is None:
            with dynamo_utils.dynamo_timed(
                "npugraphify",
                log_pt2_compile_event=True,
            ), dynamo_utils.preserve_rng_state():
                compiled_fn = npugraphify_fn(model, new_inputs, static_input_idxs)
        return compiled_fn(new_inputs)

    return run


def npugraphify_impl(
    model: Callable[..., Any],
    inputs: List[torch.Tensor],
    static_input_idxs: Sequence[int] = (),
) -> Callable[[List[InputType]], Any]:
    """
    Assumes inputs[static_input_idxs[i]] are always the same memory address
    """
    check_input_idxs = get_input_idxs_to_check(inputs, static_input_idxs)  # type: ignore[arg-type]
    static_input_idxs: OrderedSet[int] = OrderedSet(
        remove_unaligned_input_idxs(inputs, static_input_idxs)  # type: ignore[arg-type]
    )
    copy_misaligned_inputs(inputs, check_input_idxs)  # type: ignore[arg-type]

    if not isinstance(inputs, list):
        raise RuntimeError("check isinstance(inputs, list) fail")

    inps_expanded_dims = [
        get_expanded_dims(x) if idx not in static_input_idxs else []
        for idx, x in enumerate(inputs)
    ]

    # allocate static tensor inputs
    static_inputs = [
        x
        if not isinstance(x, torch.Tensor)
        else static_input(x)
        if idx not in static_input_idxs
        else x.detach()
        for idx, x in enumerate(inputs)
    ]

    # copy over input values for fresh allocations
    for idx, (x, expanded_dims) in enumerate(zip(inputs, inps_expanded_dims)):
        if isinstance(x, torch.Tensor) and idx not in static_input_idxs:
            index_expanded_dims_and_copy_(static_inputs[idx], x, expanded_dims)

    # warmup
    torch.npu.synchronize()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    # copy static_inputs because it will be cleared in model
    with torch.npu.stream(stream):
        model(list(static_inputs))
    stream.synchronize()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()

    # record
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream, capture_error_mode="thread_local"):
        static_outputs = model(list(static_inputs))
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    if config.size_asserts:

        def run(new_inputs: List[InputType]) -> Callable[[List[InputType]], Any]:
            if not len(static_inputs) == len(new_inputs):
                raise RuntimeError("check len(static_inputs) == len(new_inputs) fail")
            for idx, (dst, src, expanded_dims) in enumerate(
                zip(static_inputs, new_inputs, inps_expanded_dims)
            ):
                if not isinstance(dst, torch.Tensor):
                    continue
                if not isinstance(src, torch.Tensor):
                    raise RuntimeError("check isinstance(src, torch.Tensor) fail")
                if idx in static_input_idxs:
                    if not dst.data_ptr() == src.data_ptr():
                        raise RuntimeError("check dst.data_ptr() == src.data_ptr() fail")
                else:
                    # could make one single op of multiple slices
                    # and avoid dispatch.
                    # Could also pre-index the `dst` tensors
                    index_expanded_dims_and_copy_(dst, src, expanded_dims)
            new_inputs.clear()
            graph.replay()
            return static_outputs

    else:
        copy_indices = [
            idx 
            for idx in range(len(static_inputs))
            if idx not in static_input_idxs
        ]

        def run(new_inputs: List[InputType]) -> Callable[[List[InputType]], Any]:
            for idx in copy_indices:
                expanded_dims = inps_expanded_dims[idx]
                src = new_inputs[idx]
                if not isinstance(src, torch.Tensor):
                    raise RuntimeError("check isinstance(src, torch.Tensor) fail")
                index_expanded_dims_and_copy_(static_inputs[idx], src, expanded_dims)
            new_inputs.clear()
            graph.replay()
            return static_outputs

    return align_inputs_from_check_idxs(run, check_input_idxs)


def _apply_npugraph_tree_methods():
    torch._inductor.compile_fx.cudagraphify = npugraphify
    torch._inductor.cudagraph_utils.check_multiple_devices_or_any_cpu_nodes = check_multiple_devices_or_any_cpu_nodes
    torch.compiler.npugraph_mark_step_begin = npugraph_mark_step_begin
