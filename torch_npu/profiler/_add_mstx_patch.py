import functools
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

import torch_npu
from torch_npu.utils._step import _custom_call

original_tx_call = Module.__call__
original_iter = DataLoader.__iter__
original_save = torch.serialization.save
original_singlenext = torch.utils.data.dataloader._SingleProcessDataLoaderIter.__next__
original_multinext = torch.utils.data.dataloader._MultiProcessingDataLoaderIter.__next__


class MstxState:
    def __init__(self):
        self.module_dict = {}
        self.is_outer_call = True
        self.fp_range_id = None
        self.dataloader_range_id = None
        self.save_range_id = None

    def add_module_dict(self, module):
        self.module_dict[module] = [
            sub_module for _, sub_module in module.named_modules() if sub_module != module
        ]

    def is_child_module(self, module):
        return any(module in value for value in self.module_dict.values())

mstx_state = MstxState()


def _is_loss_module(module):
    return isinstance(module, torch.nn.modules.loss._Loss)


def _custom_tx_call(self, *args, **kwargs):
    global mstx_state

    if not torch.npu.is_initialized():
        return original_tx_call(self, *args, **kwargs)

    # the outermost module add mstx range_start
    if mstx_state.is_outer_call:
        # not the loss module and recalculation process
        if not mstx_state.is_child_module(self) and not _is_loss_module(self):
            stream = torch.npu.current_stream()
            mstx_state.fp_range_id = torch.npu.mstx.range_start("forward", stream)
            mstx_state.add_module_dict(self)
        mstx_state.is_outer_call = False
        self.tx_visited = True

    out_call = original_tx_call(self, *args, **kwargs)

    # the outermost module add mstx range_end
    if hasattr(self, "tx_visited") and self.tx_visited:
        mstx_state.is_outer_call = True
        self.tx_visited = False
        if not _is_loss_module(self) and mstx_state.fp_range_id is not None:
            torch.npu.mstx.range_end(mstx_state.fp_range_id)
            mstx_state.fp_range_id = None

    return out_call


def _custom_dataloader_iter(self):
    global mstx_state

    out_iter = original_iter(self)

    def dataloader_wrapper(func):
        def wrapper(*args, **kwargs):
            mstx_state.dataloader_range_id = torch.npu.mstx.range_start("dataloader")
            out = func(*args, **kwargs)
            if mstx_state.dataloader_range_id is not None:
                torch.npu.mstx.range_end(mstx_state.dataloader_range_id)
                mstx_state.dataloader_range_id = None
            return out

        return wrapper

    if self.num_workers == 0:
        torch.utils.data.dataloader._SingleProcessDataLoaderIter.__next__ = dataloader_wrapper(original_singlenext)
    else:
        torch.utils.data.dataloader._MultiProcessingDataLoaderIter.__next__ = dataloader_wrapper(original_multinext)

    return out_iter


def _custom_save(func):
    global mstx_state

    @functools.wraps(func)
    def save_wrapper(*args, **kwargs):
        stream = torch.npu.current_stream()
        mstx_state.save_range_id = torch.npu.mstx.range_start("save_checkpoint", stream)
        out = func(*args, **kwargs)
        if mstx_state.save_range_id is not None:
            torch.npu.mstx.range_end(mstx_state.save_range_id)
            mstx_state.save_range_id = None
        return out

    return save_wrapper


def apply_mstx_patch():
    global original_tx_call

    if Module.__call__.__name__ == "_custom_call":
        original_tx_call = _custom_call
    Module.__call__ = _custom_tx_call
    DataLoader.__iter__ = _custom_dataloader_iter
    torch.serialization.save = _custom_save(original_save)
