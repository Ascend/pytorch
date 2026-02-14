import functools
import logging
from collections import defaultdict
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
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.cudagraphs import (
    check_for_mutation_ignore_cuda_graph_managed_tensor,
    find_input_mutations,
    get_device_node_mapping,
    get_stack_traces,
)
from torch._dynamo.backends.debugging import boxed_nop
from torch._dynamo.backends.registry import register_backend
from torch._inductor import config
from torch._inductor.compile_fx import (
    get_input_idxs_to_check,
    index_expanded_dims_and_copy_,
    static_input,
)
from torch._inductor.cudagraph_utils import (
    _get_use_stack_trace,
    format_default_skip_message,
    get_mutation_stack_trace,
    get_placeholder_info,
    log_cudagraph_skip_and_bump_counter,
    BoxedDeviceIndex,
    PlaceholderInfo,
)
from torch._inductor.output_code import get_expanded_dims
from torch._inductor.utils import (
    align_inputs_from_check_idxs,
    copy_misaligned_inputs,
    count_tangents,
    get_first_incompatible_cudagraph_node,
    num_fw_fixed_arguments,
    output_node,
    remove_unaligned_input_idxs,
    BoxedBool,
    InputType,
)
from torch.multiprocessing.reductions import StorageWeakRef
import torch_npu.npu.aclnn


log = logging.getLogger("torch_npu.aclgraph")


def npugraph_mark_step_begin():
    from torch_npu.npu._graph_tree import mark_step_begin
    mark_step_begin()


def check_multiple_devices_or_any_cpu_nodes(
    device_node_mapping: Dict[torch.device, torch.fx.Node]
) -> Optional[str]:
    from torch_npu._inductor import config as npu_config
    if npu_config.npugraph_trees.disable_cpu_input_check:
        device_node_mapping.pop(torch.device("cpu"), None)

    cpu_node = device_node_mapping.get(torch.device("cpu"))
    if cpu_node:
        msg = f"cpu device ({cpu_node.name})"
        stack_trace = _get_use_stack_trace(cpu_node)
        log.info(f"skip with cpu node, msg is {msg}, stack_trace is {stack_trace}")
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
    if torch_npu.npu.aclnn._use_static_aclnn_kernel:
        from torch_npu._inductor.npu_static_kernel import StaticKernelCompiler
        static_kernel_complier = StaticKernelCompiler()
        with static_kernel_complier:
            with torch.npu.stream(stream):
                model(list(static_inputs))
    else:
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


def check_for_skip(aot_model: torch.fx.GraphModule, num_fixed) -> Optional[str]:
    if not torch._dynamo.config.cudagraph_backend_support_input_mutation:
        mut_skip = check_for_mutation_ignore_cuda_graph_managed_tensor(
            aot_model, num_fixed
        )
        if mut_skip:
            return mut_skip

    skip = check_multiple_devices_or_any_cpu_nodes(
        get_device_node_mapping(aot_model)
    )
    if skip:
        return skip

    node = get_first_incompatible_cudagraph_node(aot_model)
    if node:
        return format_default_skip_message(f"incompatible op ({node.name})")

    return None


def get_device_index(gm) -> int:
    device_node_mapping = get_device_node_mapping(gm)
    from torch_npu._inductor import config as npu_config
    if npu_config.npugraph_trees.disable_cpu_input_check:
        device_node_mapping.pop(torch.device("cpu"), None)
    device = next(iter(device_node_mapping))
    if not (device.type == "npu"):
        raise RuntimeError("check device.type == npu fail", )
    return device.index


def npugraphs(dynamo_model, dynamo_inputs):
    from torch_npu.npu._graph_tree import npugraphify_impl as new_npugraphify_impl

    do_npugraphs = BoxedBool(True)
    boxed_device_index = BoxedDeviceIndex(None)

    def forward_npugraphs(aot_model, aot_inputs, is_inference=False):
        interp = boxed_nop(aot_model, aot_inputs)
        fixed = num_fw_fixed_arguments(len(dynamo_inputs), len(aot_inputs))
        skip_msg = check_for_skip(aot_model, fixed)
        if skip_msg:
            BoxedBool.disable(do_npugraphs)
            log_cudagraph_skip_and_bump_counter(
                f"skipping npugraphs due to {skip_msg}"
            )
            return interp

        boxed_device_index.set(get_device_index(aot_model))
        out = new_npugraphify_impl(
            interp,
            aot_inputs,
            range(fixed),
            device_index=boxed_device_index.value,
            is_backward=False,
            is_inference=False,
            stack_traces=get_stack_traces(aot_model),
            placeholders=get_placeholder_info(aot_model.graph),
            mutated_input_idxs=find_input_mutations(aot_model.graph),
        )
        out._boxed_call = True
        return out

    def backward_npugraphs(aot_model, aot_inputs):
        interp = boxed_nop(aot_model, aot_inputs)
        if not do_npugraphs:
            return aot_model

        fixed = count_tangents(aot_model)

        skip_msg = check_for_skip(aot_model, fixed)
        if skip_msg:
            log_cudagraph_skip_and_bump_counter(
                "skipping npugraphs due to %s", skip_msg
            )

            # See [Backward Generation Handling]
            from torch_npu.npu._graph_tree import get_manager
            manager = get_manager(
                boxed_device_index.value, create_if_none_exists=False
            )

            if manager is None:
                raise RuntimeError("check manager is None fail")

            def fn(inputs):
                manager.set_to_running_backward()
                return aot_model(inputs)

            fn._boxed_call = True
            return fn

        out = new_npugraphify_impl(
            interp,
            aot_inputs,
            range(fixed),
            device_index=get_device_index(aot_model),
            is_backward=True,
            is_inference=False,
            stack_traces=get_stack_traces(aot_model),
            placeholders=get_placeholder_info(aot_model.graph),
            mutated_input_idxs=find_input_mutations(aot_model.graph),
        )
        out._boxed_call = True
        return out

    aot_npugraphs = aot_autograd(
        fw_compiler=forward_npugraphs,
        bw_compiler=backward_npugraphs,
        inference_compiler=functools.partial(forward_npugraphs, is_inference=True),
        keep_inference_input_mutations=torch._dynamo.config.cudagraph_backend_keep_input_mutation,
    )
    return aot_npugraphs(dynamo_model, dynamo_inputs)


class NpugraphsBackend:
    compiler_name = "npugraphs"

    @staticmethod
    def reset():
        from torch_npu.npu._graph_tree import reset_npugraph_trees

        reset_npugraph_trees()

    @staticmethod
    def __call__(model, inputs):
        return npugraphs(model, inputs)


def _apply_npugraph_tree_methods():
    # aot_npugraphs only applies graphs to the graph.  It is also helpful
    # for debugging and can serve as a perf baseline.
    register_backend(name="npugraphs", compiler_fn=NpugraphsBackend())
    torch._inductor.compile_fx.cudagraphify = npugraphify
    torch._inductor.cudagraph_utils.check_multiple_devices_or_any_cpu_nodes = check_multiple_devices_or_any_cpu_nodes
    torch.compiler.npugraph_mark_step_begin = npugraph_mark_step_begin
