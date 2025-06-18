__all__ = ["is_current_stream_capturing", "graph_pool_handle", "graph_task_group_begin",
           "graph_task_group_end", "graph_task_update_begin", "graph_task_update_end",
           "NPUGraph", "graph", "make_graphed_callables"]

import gc
import re
import typing
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch_npu._C
from torch_npu._C import _weak_ref_tensor as TensorWeakRef
from torch_npu.utils._error_code import ErrCode, pta_error
from .utils import _dummy_type


if not hasattr(torch_npu._C, "_NPUStreamBase"):
    # Define dummy base classes
    torch_npu._C.__dict__["_NPUGraph"] = _dummy_type("_NPUGraph")
    torch_npu._C.__dict__["_graph_pool_handle"] = _dummy_type("_graph_pool_handle")
    torch_npu._C.__dict__["_npu_isCurrentStreamCapturing"] = _dummy_type(
        "_npu_isCurrentStreamCapturing"
    )
    torch_npu._C.__dict__["_graph_task_group_begin"] = _dummy_type("_graph_task_group_begin")
    torch_npu._C.__dict__["_graph_task_group_end"] = _dummy_type("_graph_task_group_end")
    torch_npu._C.__dict__["_graph_task_update_begin"] = _dummy_type("_graph_task_update_begin")
    torch_npu._C.__dict__["_graph_task_update_end"] = _dummy_type("_graph_task_update_end")

from torch_npu._C import (  # noqa: F401
    _npu_isCurrentStreamCapturing,
    _NPUGraph,
    _graph_pool_handle,
    _graph_task_group_begin,
    _graph_task_group_end,
    _graph_task_update_begin,
    _graph_task_update_end,
)


def is_current_stream_capturing():
    r"""Return True if NPU graph capture is underway on the current NPU stream, False otherwise.

    If a NPU context does not exist on the current device, returns False without initializing the context.
    """
    return _npu_isCurrentStreamCapturing()


# Python shim helps Sphinx process docstrings more reliably.
def graph_pool_handle():
    r"""Return an opaque token representing the id of a graph memory pool.

    See :ref:`Graph memory management<graph-memory-management>`.

    .. warning::
        This API is in beta and may change in future releases.
    """
    return _graph_pool_handle()


def graph_task_group_begin(stream):
    _graph_task_group_begin(stream)


def graph_task_group_end(stream):
    return _graph_task_group_end(stream)


def graph_task_update_begin(stream, handle):
    _graph_task_update_begin(stream, handle)


def graph_task_update_end(stream):
    _graph_task_update_end(stream)


@dataclass
class _GraphDispatchRecord:
    """存储单次操作的完整记录"""
    event: Any = None
    handle: Any = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    args: Tuple[Any, ...] = field(default_factory=tuple)
    op_cache_entry: Any = None


class _GraphDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):
    tensor_schema_name = {}
    update_stream = None

    def __init__(self):
        self.graph_dispatch_records = []
        if not self.update_stream:
            self.update_stream = torch_npu.npu.Stream()

    @classmethod
    def update_schema(cls, name, schame):
        if name in cls.tensor_schema_name:
            return
        pattern = r'(?:Tensor\??\s*)(\w+)'
        cls.tensor_schema_name[name] = re.findall(pattern, schame)
    
    def update_capture_record(self, cpu_update_input):
        if len(cpu_update_input) == 1:
            new_list = [cpu_update_input[0].copy() for _ in range(len(self.graph_dispatch_records))]
            cpu_update_input = new_list 
        if len(self.graph_dispatch_records) != len(self.graph_dispatch_records):
            raise RuntimeError(f"Currently, there are {len(self.graph_dispatch_records)} operators that need to be updated by capture, "
                f"and there are only {len(self.graph_dispatch_records)} elements in the incoming cpu_update_input list", pta_error(ErrCode.PARAM))
        with torch.npu.stream(self.update_stream):
            for graph_dispatch_record, update_input in zip(self.graph_dispatch_records, cpu_update_input):
                graph_task_update_begin(self.update_stream, graph_dispatch_record.handle)
                for key in update_input:
                    graph_dispatch_record.kwargs[key] = update_input[key]
                graph_dispatch_record.op_cache_entry(*graph_dispatch_record.args, **graph_dispatch_record.kwargs)
                graph_task_update_end(self.update_stream)
                graph_dispatch_record.event.record(self.update_stream)

    def _append_dispatch_record(self, event, handle, args, kwargs, func):
        args_ref = []
        for element in args:
            if torch.is_tensor(element):
                args_ref.append(TensorWeakRef(element))
            else:
                args_ref.append(deepcopy(element))
        kwargs_ref = {}
        for key, vaule in kwargs.items():
            if key == "out":
                kwargs_ref[key] = [TensorWeakRef(vaule[0]), TensorWeakRef(vaule[1])]
            elif key in self.tensor_schema_name[str(func.__name__)]:
                kwargs_ref[key] = TensorWeakRef(vaule)
            else:
                kwargs_ref[key] = deepcopy(vaule)
        return _GraphDispatchRecord(event=event, handle=handle, kwargs=kwargs_ref, args=tuple(args_ref), op_cache_entry=func)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.__name__ == "npu_fused_infer_attention_score":
            func_out = torch_npu.npu_fused_infer_attention_score.out
            self.update_schema(str(func_out.__name__), str(func_out._schema))
            stream = torch_npu.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            # apply tensor
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(*args, **kwargs)
            output = torch.empty_like(args[0])
            softmax_lse = torch.empty(1, dtype=args[0].dtype, device=args[0].device)
            kwargs["workspace"] = workspace
            kwargs["out"] = [output, softmax_lse]
            # begin graph task
            graph_task_group_begin(stream)
            func_out(*args, **kwargs)
            handle = graph_task_group_end(stream)
            # save state for update
            self.graph_dispatch_records.append(
                self._append_dispatch_record(event, handle, args, kwargs, func_out))
            return kwargs["out"]
        elif func.__name__ == "npu_fused_infer_attention_score.out":
            self.update_schema(str(func.__name__), str(func._schema))
            stream = torch_npu.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            # begin graph task
            graph_task_group_begin(stream)
            func(*args, **kwargs)
            handle = graph_task_group_end(stream)
            # save state for update
            self.graph_dispatch_records.append(
                self._append_dispatch_record(event, handle, args, kwargs, func))
            return kwargs["out"]
        else:
            return func(*args, **kwargs)


# Python shim helps Sphinx process docstrings more reliably.
class NPUGraph(torch_npu._C._NPUGraph):
    r"""Wrapper around a NPU graph.

    .. warning::
        This API is in beta and may change in future releases.
    """

    def __new__(cls):
        return super().__new__(cls)
    
    def __init__(self):
        self.graph_dispatch_mode = _GraphDispatchMode()
        self.auto_dispatch_capture = False
        return super().__init__()

    def capture_begin(self, pool=None, capture_error_mode="global"):
        r"""Begin capturing NPU work on the current stream.

        Typically, you shouldn't call ``capture_begin`` yourself.
        Use :class:`~torch.npu.graph` or :func:`~torch.npu.make_graphed_callables`,
        which call ``capture_begin`` internally.

        Arguments:
            pool (optional): Token (returned by :func:`~torch.npu.graph_pool_handle` or
                :meth:`other_Graph_instance.pool()<torch.npu.NPUGraph.pool>`) that hints this graph may share memory
                with the indicated pool.  See :ref:`Graph memory management<graph-memory-management>`.
            capture_error_mode (str, optional): specifies the aclmdlRICaptureMode for the graph capture stream.
                Can be "global", "thread_local" or "relaxed". During npu graph capture, some actions, such as npuMalloc,
                may be unsafe. "global" will error on actions in other threads, "thread_local" will only error for
                actions in the current thread, and "relaxed" will not error on these actions. Do NOT change this setting
                unless you're familiar with `aclmdlRICaptureMode`_
        """  # noqa: B950
        super().capture_begin(pool=pool, capture_error_mode=capture_error_mode)

    def capture_end(self):
        r"""End NPU graph capture on the current stream.

        After ``capture_end``, ``replay`` may be called on this instance.

        Typically, you shouldn't call ``capture_end`` yourself.
        Use :class:`~torch.npu.graph` or :func:`~torch.npu.make_graphed_callables`,
        which call ``capture_end`` internally.
        """
        super().capture_end()

    def replay(self):
        r"""Replay the NPU work captured by this graph."""
        super().replay()

    def reset(self):
        r"""Delete the graph currently held by this instance."""
        super().reset()

    def pool(self):
        r"""Return an opaque token representing the id of this graph's memory pool.

        This id can optionally be passed to another graph's ``capture_begin``,
        which hints the other graph may share the same memory pool.
        """
        return super().pool()

    def update(self, cpu_update_input):
        if not self.auto_dispatch_capture:
            raise RuntimeError("The current graph configuration does not support update,"
                "Try to capture by setting auto_dispatch_capture=True during capture", pta_error(ErrCode.PARAM))
        self.graph_dispatch_mode.update_capture_record(cpu_update_input)


class graph:
    r"""Context-manager that captures NPU work into a :class:`torch.npu.NPUGraph` object for later replay.

    See :ref:`CUDA Graphs <cuda-graph-semantics>` for a general introduction,
    detailed use, and constraints.

    Arguments:
        npu_graph (torch.npu.NPUGraph): Graph object used for capture.
        pool (optional): Opaque token (returned by a call to :func:`~torch.npu.graph_pool_handle()` or
            :meth:`other_Graph_instance.pool()<torch.npu.NPUGraph.pool>`) hinting this graph's capture
            may share memory from the specified pool. See :ref:`Graph memory management<graph-memory-management>`.
        stream (torch.npu.Stream, optional): If supplied, will be set as the current stream in the context.
            If not supplied, ``graph`` sets its own internal side stream as the current stream in the context.
        capture_error_mode (str, optional): specifies the aclmdlRICaptureMode for the graph capture stream.
            Can be "global", "thread_local" or "relaxed". During npu graph capture, some actions, such as npuMalloc,
            may be unsafe. "global" will error on actions in other threads, "thread_local" will only error for
            actions in the current thread, and "relaxed" will not error on actions. Do NOT change this setting
            unless you're familiar with `aclmdlRICaptureMode`_

    .. note::
        For effective memory sharing, if you pass a ``pool`` used by a previous capture and the previous capture
        used an explicit ``stream`` argument, you should pass the same ``stream`` argument to this capture.

    .. warning::
        This API is in beta and may change in future releases.
    """  # noqa: B950

    default_capture_stream: typing.Optional["torch.npu.Stream"] = None

    def __init__(
        self,
        npu_graph,
        pool=None,
        stream=None,
        auto_dispatch_capture=False,
        capture_error_mode: str = "global",
    ):
        # Lazy-init of default_capture_stream helps avoid circular-import errors.
        # Not thread safe, but graphs already have the general (explicitly documented)
        # restriction that only one capture may be underway at a time in the process.
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.npu.Stream()

        self.pool = () if pool is None else (pool,)
        self.capture_stream = (
            stream if stream is not None else self.__class__.default_capture_stream
        )
        if self.capture_stream is None:
            raise RuntimeError("capture stream is None")
        self.stream_ctx = torch.npu.stream(self.capture_stream)
        self.npu_graph = npu_graph
        self.capture_error_mode = capture_error_mode
        self.npu_graph.auto_dispatch_capture = auto_dispatch_capture

    def __enter__(self):
        # Free as much memory as we can for the graph
        torch.npu.synchronize()
        gc.collect()
        torch.npu.empty_cache()

        # Stackoverflow seems comfortable with this pattern
        self.stream_ctx.__enter__()
        if self.npu_graph.auto_dispatch_capture:
            self.npu_graph.graph_dispatch_mode.__enter__()
        self.npu_graph.capture_begin(
            *self.pool, capture_error_mode=self.capture_error_mode
        )

    def __exit__(self, exc_type, exc_value, traceback):
        self.npu_graph.capture_end()
        if self.npu_graph.auto_dispatch_capture:
            self.npu_graph.graph_dispatch_mode.__exit__(exc_type, exc_value, traceback)
        self.stream_ctx.__exit__(exc_type, exc_value, traceback)
        # returning None should propagate exceptions from either capture_end or stream_ctx.__exit__()


def make_graphed_callables(
        callables, sample_args, num_warmup_iters=3, allow_unused_input=False, pool=None
):
    r"""Accept callables (functions or :class:`nn.Module<torch.nn.Module>`\ s) and returns graphed versions.

    Each graphed callable's forward pass runs its source callable's
    forward CUDA work as a CUDA graph inside a single autograd node.

    The graphed callable's forward pass also appends
    a backward node to the autograd graph. During backward, this node runs the
    callable's backward work as a CUDA graph.

    Therefore, each graphed callable should be a drop-in replacement for its source callable
    in an autograd-enabled training loop.

    See :ref:`Partial-network capture<partial-network-capture>` for detailed use and constraints.

    If you pass a tuple of several callables, their captures will use the same memory pool.
    See :ref:`Graph memory management<graph-memory-management>` for when this is appropriate.

    Arguments:
        callables (torch.nn.Module or Python function, or tuple of these): Callable or callables to graph.
            See :ref:`Graph memory management<graph-memory-management>` for when passing a tuple of callables
            is appropriate.  If you pass a tuple of callables, their order in the tuple must be the same order
            they'll run in the live workload.
        sample_args (tuple of Tensors, or tuple of tuples of Tensors): Samples args for each callable.
            If a single callable was passed, ``sample_args`` must be a single tuple of argument Tensors.
            If a tuple of callables was passed, ``sample_args`` must be tuple of tuples of argument Tensors.
        num_warmup_iters (int): The number of warmup iterations. Currently, ``DataDistributedParallel`` needs
            11 iterations for warm up. Default: ``3``.
        allow_unused_input (bool): If False, specifying inputs that were not used when computing outputs
            (and therefore their grad is always zero) is an error. Defaults to False.
        pool (optional): Token (returned by :func:`~torch.cuda.graph_pool_handle` or
            :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) that hints this graph may share memory
            with the indicated pool.  See :ref:`Graph memory management<graph-memory-management>`.
    .. note::
        The ``requires_grad`` state of each Tensor in ``sample_args`` must match the state
        that's expected for the corresponding real input in the training loop.

    .. warning::
        This API is in beta and may change in future releases.

    .. warning::
        ``sample_args`` for each callable must contain only Tensors. Other types are not allowed.

    .. warning::
        Returned callables do not support higher order differentiation (e.g., double backward).

    .. warning::
        In any :class:`~torch.nn.Module` passed to :func:`~make_graphed_callables`, only parameters
        may be trainable. Buffers must have ``requires_grad=False``.

    .. warning::
        After you pass a :class:`torch.nn.Module` through :func:`~make_graphed_callables`,
        you may not add or remove any of that Module's parameters or buffers.

    .. warning::
        :class:`torch.nn.Module`\s passed to :func:`~torch.cuda.make_graphed_callables` must not have module hooks
        registered on them at the time they are passed. However, registering hooks on modules *after* passing them
        through :func:`~torch.cuda.make_graphed_callables` is allowed.

    .. warning::
        When running a graphed callable, you must pass its arguments in the same order and format
        they appeared in that callable's ``sample_args``.

    .. warning::
        The automatic mixed precision is supported in :func:`~torch.cuda.make_graphed_callables` only with disabled
        caching. The context manager `torch.cuda.amp.autocast()` must have `cache_enabled=False`.
    """
    if torch_npu.npu.is_autocast_enabled() and torch.is_autocast_cache_enabled():
        raise RuntimeError(
            "make_graphed_callables does not support the autocast caching. Please set `cache_enabled=False`."
        )

    just_one_callable = False

    if not isinstance(callables, tuple):
        just_one_callable = True
        callables = (callables,)
        sample_args = (sample_args,)

    flatten_sample_args = []

    for c, args in zip(callables, sample_args):
        if isinstance(c, torch.nn.Module):
            if len(c._backward_hooks) > 0 or len(c._forward_hooks) > 0 or len(c._forward_pre_hooks) > 0:
                raise RuntimeError("Modules must not have hooks registered at the time they are passed. However, "
                    + "registering hooks on modules after passing them through make_graphed_callables is allowed.")
            if any(b.requires_grad for b in c.buffers()):
                raise RuntimeError("In any :class:`~torch.nn.Module` passed to :func:`~make_graphed_callables`,"
                    + " only parameters may be trainable. All buffers must have ``requires_grad=False``.")
        flatten_arg = torch.utils._pytree.arg_tree_leaves(*args)
        flatten_sample_args.append(tuple(flatten_arg))
        if not all(isinstance(arg, torch.Tensor) for arg in flatten_arg):
            raise RuntimeError("In the beta API, sample_args "
                + "for each callable must contain only Tensors. Other types are not allowed.")

    # If a callable is an nn.Module, its graph's full input surface is the args the user explicitly
    # passes to forward (ie, its sample_args) AND the module's parameter attributes.
    per_callable_len_user_args = [len(args) for args in flatten_sample_args]
    per_callable_module_params = [
        tuple(c.parameters()) if isinstance(c, torch.nn.Module) else ()
        for c in callables
    ]
    per_callable_static_input_surfaces = [
        flatten_sample_args[i] + per_callable_module_params[i]
        for i in range(len(callables))
    ]

    fwd_graphs = [torch_npu.npu.NPUGraph() for _ in range(len(callables))]
    bwd_graphs = [torch_npu.npu.NPUGraph() for _ in range(len(callables))]

    mempool = graph_pool_handle() if pool is None else pool

    # Warmup
    # Hopefully prevents cudnn benchmarking and other lazy-initialization cuda work
    # from ending up in any captures.
    torch_npu.npu.synchronize()
    with torch_npu.npu.stream(torch_npu.npu.Stream()):
        for func, args, static_input_surface in zip(
            callables, sample_args, per_callable_static_input_surfaces
        ):
            grad_inputs, outputs, outputs_grad = None, None, None
            for _ in range(num_warmup_iters):
                outputs = torch.utils._pytree.tree_leaves(func(*args))
                outputs_grad = tuple(o for o in outputs if o.requires_grad)
                if len(outputs_grad) > 0:
                    grad_inputs = torch.autograd.grad(
                        outputs=outputs_grad,
                        inputs=tuple(
                            i for i in static_input_surface if i.requires_grad
                        ),
                        grad_outputs=tuple(
                            torch.empty_like(o) for o in outputs if o.requires_grad
                        ),
                        only_inputs=True,
                        allow_unused=allow_unused_input,
                    )
            for v in [outputs, outputs_grad, grad_inputs]:
                del v

    torch_npu.npu.synchronize()

    # All captures here share a mempool. To avoid replays corrupting each other's memory,
    # the safest approach is to capture all passes in the same order they'll run:
    # fwd 1, fwd 2, ... fwd N, then bwd N, bwd N-1, ... bwd 1.

    # Capture forward graphs
    per_callable_static_outputs = []
    per_callable_output_unflatten_spec = []
    for func, args, fwd_graph in zip(callables, sample_args, fwd_graphs):
        with torch_npu.npu.graph(fwd_graph, pool=mempool):
            outputs = func(*args)

        flatten_outputs, spec = torch.utils._pytree.tree_flatten(outputs)
        per_callable_static_outputs.append(tuple(flatten_outputs))
        per_callable_output_unflatten_spec.append(spec)

    # Capture backward graphs in reverse order
    per_callable_static_grad_outputs = []
    per_callable_static_grad_inputs = []
    for static_input_surface, static_outputs, bwd_graph, module_params in zip(
        reversed(per_callable_static_input_surfaces),
        reversed(per_callable_static_outputs),
        reversed(bwd_graphs),
        reversed(per_callable_module_params),
    ):
        # For now, assumes all static_outputs require grad
        static_grad_outputs = tuple(
            torch.empty_like(o) if o.requires_grad else None for o in static_outputs
        )

        outputs_grad = tuple(o for o in static_outputs if o.requires_grad)
        grad_inputs = None
        if len(outputs_grad) > 0:
            with torch_npu.npu.graph(bwd_graph, pool=mempool):
                grad_inputs = torch.autograd.grad(
                    outputs=outputs_grad,
                    inputs=tuple(i for i in static_input_surface if i.requires_grad),
                    grad_outputs=tuple(o for o in static_grad_outputs if o is not None),
                    only_inputs=True,
                    allow_unused=allow_unused_input,
                )

        # Constructs a tuple suitable for returning from Graphed.backward:
        # Pads out the actually-needed grads with Nones in gradient slots for inputs that don't require grad.
        # I couldn't think of a slick one-liner for this pattern.
        static_grad_inputs = []
        grad_idx = 0
        for arg in static_input_surface:
            if arg.requires_grad and grad_inputs is not None:
                static_grad_inputs.append(grad_inputs[grad_idx])
                grad_idx += 1
            else:
                static_grad_inputs.append(None)  # type: ignore[arg-type]
        static_grad_inputs = tuple(static_grad_inputs)  # type: ignore[assignment]

        per_callable_static_grad_outputs.append(static_grad_outputs)
        per_callable_static_grad_inputs.append(static_grad_inputs)

    # Reverses the most recent two lists
    per_callable_static_grad_outputs.reverse()
    per_callable_static_grad_inputs.reverse()
    # Now for every per_callable list, per_callable_*[i] holds the stuff for the ith callable.

    def make_graphed_autograd_function(
        fwd_graph,
        bwd_graph,
        module_params,
        len_user_args,
        output_unflatten_spec,
        static_input_surface,
        static_outputs,
        static_grad_outputs,
        static_grad_inputs,
    ):
        class Graphed(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *inputs):
                # At this stage, only the user args may (potentially) be new tensors.
                for i in range(len_user_args):
                    if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                        static_input_surface[i].copy_(inputs[i])
                fwd_graph.replay()
                if not isinstance(static_outputs, tuple):
                    raise RuntimeError("static_outputs is not tuple.")
                return tuple(o.detach() for o in static_outputs)

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, *grads):
                if (len(grads) != len(static_grad_outputs)):
                    raise RuntimeError("The length of grads"
                        + " is not equal with the length of static_grad_outputs.")
                for g, grad in zip(static_grad_outputs, grads):
                    if g is not None:
                        # don't copy if autograd gods have been kind and the
                        # incoming grad is already in the right place
                        if g.data_ptr() != grad.data_ptr():
                            g.copy_(grad)
                bwd_graph.replay()

                # Input args that didn't require grad expect a None gradient.
                if not isinstance(static_grad_inputs, tuple):
                    raise RuntimeError("static_grad_inputs is not tuple.")
                return tuple(
                    b.detach() if b is not None else b for b in static_grad_inputs
                )

        def functionalized(*user_args):
            # Runs the autograd function with inputs == all inputs to the graph that might require grad
            # (explicit user args + module parameters)
            # Assumes module params didn't change since capture.
            flatten_user_args = torch.utils._pytree.arg_tree_leaves(*user_args)
            out = Graphed.apply(*(tuple(flatten_user_args) + module_params))
            return torch.utils._pytree.tree_unflatten(out, output_unflatten_spec)

        return functionalized

    # Put together the final graphed callables
    ret = []
    for i, func in enumerate(callables):
        graphed = make_graphed_autograd_function(
            fwd_graphs[i],
            bwd_graphs[i],
            per_callable_module_params[i],
            per_callable_len_user_args[i],
            per_callable_output_unflatten_spec[i],
            per_callable_static_input_surfaces[i],
            per_callable_static_outputs[i],
            per_callable_static_grad_outputs[i],
            per_callable_static_grad_inputs[i],
        )

        if isinstance(func, torch.nn.Module):

            def make_graphed_forward(func, graph_training_state, graphed, orig_fwd):
                def new_fwd(*user_args):
                    # If the module's training-or-eval state matches what we graphed,
                    # run the graph, otherwise run the original forward method
                    if func.training == graph_training_state:
                        return graphed(*user_args)
                    else:
                        return orig_fwd(*user_args)

                return new_fwd

            func.forward = make_graphed_forward(func, func.training, graphed, func.forward)  # type: ignore[assignment]
            ret.append(func)
        else:
            ret.append(graphed)

    if just_one_callable:
        return ret[0]

    return tuple(ret)