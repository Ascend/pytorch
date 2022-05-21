import torch_npu
from .utils import _lazy_init


def enable_graph_mode():
    return torch_npu._C._npu_enable_graph_mode()


def disable_graph_mode():
    _lazy_init()
    return torch_npu._C._npu_disable_graph_mode()


def is_graph_mode() -> bool:
    if not hasattr(torch_npu._C, "_npu_is_graph_mode"):
        return False
    return torch_npu._C._npu_is_graph_mode()


def launch_graph():
    _lazy_init()
    if not is_graph_mode():
        raise RuntimeError("Npu run mode must be graph mode when launch graph")
    return torch_npu._C._npu_launch_graph()