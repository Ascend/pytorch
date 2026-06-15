from collections.abc import Generator
from contextlib import contextmanager

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


__all__ = ["NPUGraphCaptureControlFlowOpDispatchMode", "ControlFlowOpWarmupDispatchMode"]


class NPUGraphCaptureControlFlowOpDispatchMode(TorchDispatchMode):
    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def __init__(self) -> None:
        self.supports_higher_order_operators = True
        super().__init__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.higher_order.cond:
            with self:
                return if_else_node(*args)
        kwargs = {} if kwargs is None else kwargs
        return func(*args, **kwargs)


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
        if func is torch.ops.higher_order.cond:
            if torch.npu.is_current_stream_capturing():
                with self:
                    return if_else_node(*args)
            with (
                torch.npu.graph(
                    torch.npu.NPUGraph(),
                    pool=None,
                    stream=self.capture_stream,
                    capture_error_mode="relaxed",
                ),
                self,
            ):
                if_else_node(*args)
            return func(*args, **kwargs)
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
