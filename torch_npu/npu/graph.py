import torch_npu
from .utils import _lazy_init


def enable_graph_mode():
    torch_npu._C._npu_enable_graph_mode()


def disable_graph_mode():
    _lazy_init()
    torch_npu._C._npu_disable_graph_mode()


def is_graph_mode() -> bool:
    return torch_npu._C._npu_is_graph_mode()


def launch_graph():
    _lazy_init()
    if not is_graph_mode():
        raise RuntimeError("Npu run mode must be graph mode when launch graph")
    torch_npu._C._npu_launch_graph()