from collections.abc import Generator
from contextlib import contextmanager

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

__all__ = ["NPUGraphCaptureControlFlowOpDispatchMode", "ControlFlowOpWarmupDispatchMode"]


def _get_capture_fn(func):
    if func is torch.ops.higher_order.cond:
        return if_else_node
    if func is getattr(torch.ops.higher_order, "while_loop", None) and hasattr(
        torch.npu.NPUGraph, "begin_capture_to_while_node"
    ):
        return while_loop_node
    return None


class NPUGraphCaptureControlFlowOpDispatchMode(TorchDispatchMode):
    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def __init__(self) -> None:
        self.supports_higher_order_operators = True
        super().__init__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        capture_fn = _get_capture_fn(func)
        if capture_fn is None:
            return func(*args, **(kwargs or {}))
        with self:
            return capture_fn(*args)


class ControlFlowOpWarmupDispatchMode(TorchDispatchMode):
    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def __init__(self) -> None:
        super().__init__()
        self.supports_higher_order_operators = True
        self.capture_stream = torch.npu.Stream()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        capture_fn = _get_capture_fn(func)
        if capture_fn is None:
            return func(*args, **kwargs)
        if torch.npu.is_current_stream_capturing():
            with self:
                return capture_fn(*args)
        with (
            torch.npu.graph(
                torch.npu.NPUGraph(),
                pool=None,
                stream=self.capture_stream,
                capture_error_mode="relaxed",
            ),
            self,
        ):
            capture_fn(*args)
        return func(*args, **kwargs)


@contextmanager
def _if_body(pred: torch.Tensor) -> Generator[None, None, None]:
    current_npu_graph = torch.npu.NPUGraph.get_currently_capturing_graph()
    current_npu_graph.begin_capture_to_if_node(pred)
    try:
        yield
    finally:
        current_npu_graph.end_capture_to_conditional_node()


def if_else_node(pred: torch.Tensor, true_fn, false_fn, operands):
    if not pred.is_npu:
        raise ValueError(
            "Conditions must be on an npu device to use conditional nodes in npu graphs"
        )

    outs = []
    for lazy_pred, fn in [
        (lambda: pred, true_fn),
        (lambda: torch.logical_not(pred), false_fn),
    ]:
        with _if_body(lazy_pred()):
            outs.append(fn(*operands))
            if len(outs) == 2:
                for if_out, else_out in zip(
                    pytree.tree_iter(outs[0]), pytree.tree_iter(outs[1])
                ):
                    if_out.copy_(else_out)
    return outs[0]


def _is_boolean_scalar_npu_tensor(pred: object) -> bool:
    return (
        isinstance(pred, torch.Tensor)
        and pred.size() == torch.Size([])
        and pred.dtype == torch.bool
        and pred.is_npu
    )


@contextmanager
def _while_body(pred: torch.Tensor):
    graph = torch.npu.NPUGraph.get_currently_capturing_graph()
    graph.begin_capture_to_while_node(pred)
    try:
        yield graph
    finally:
        graph.end_capture_to_conditional_node()


def while_loop_node(cond_fn, body_fn, carried_inputs, additional_inputs):
    flat_carried_inputs, carried_spec = pytree.tree_flatten(carried_inputs)
    if not all(isinstance(inp, torch.Tensor) for inp in flat_carried_inputs):
        raise RuntimeError(
            "NPU graph while_loop conditional nodes only support tensor carried_inputs"
        )

    flat_loop_carried = [inp.clone() for inp in flat_carried_inputs]
    loop_carried = pytree.tree_unflatten(flat_loop_carried, carried_spec)

    pred = cond_fn(*loop_carried, *additional_inputs)
    if not _is_boolean_scalar_npu_tensor(pred):
        raise RuntimeError(
            f"cond_fn must return a boolean scalar NPU tensor but got {pred}"
        )

    with _while_body(pred) as current_npu_graph:
        body_out = body_fn(*loop_carried, *additional_inputs)
        flat_body_out, body_out_spec = pytree.tree_flatten(body_out)
        if body_out_spec != carried_spec:
            raise RuntimeError(
                "body_fn should return the same pytree structure as carried_inputs"
            )
        if not all(isinstance(out, torch.Tensor) for out in flat_body_out):
            raise RuntimeError(
                "NPU graph while_loop conditional nodes only support tensor "
                "body_fn outputs"
            )

        for carried, out in zip(flat_loop_carried, flat_body_out):
            if carried.data_ptr() != out.data_ptr():
                carried.copy_(out)

        pred = cond_fn(*loop_carried, *additional_inputs)
        if not _is_boolean_scalar_npu_tensor(pred):
            raise RuntimeError(
                f"cond_fn must return a boolean scalar NPU tensor but got {pred}"
            )
        current_npu_graph.set_conditional_handle_for_current_node(pred)

    return loop_carried
