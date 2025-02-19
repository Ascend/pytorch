import functools
import torch
from torch.utils.data import DataLoader
import torch_npu

original_save = torch.serialization.save
original_iter = DataLoader.__iter__
original_singlenext = torch.utils.data.dataloader._SingleProcessDataLoaderIter.__next__
original_multinext = torch.utils.data.dataloader._MultiProcessingDataLoaderIter.__next__


class _MstxState:
    def __init__(self):
        self.dataloader_range_id = None
        self.save_range_id = None

mstx_state = _MstxState()


def _custom_dataloader_iter(self):
    global mstx_state

    out_iter = original_iter(self)

    def dataloader_wrapper(func):
        def wrapper(*args, **kwargs):
            mstx_state.dataloader_range_id = torch_npu.npu.mstx.range_start("dataloader")
            out = func(*args, **kwargs)
            if mstx_state.dataloader_range_id is not None:
                torch_npu.npu.mstx.range_end(mstx_state.dataloader_range_id)
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
        mstx_state.save_range_id = torch_npu.npu.mstx.range_start("save_checkpoint")
        out = func(*args, **kwargs)
        if mstx_state.save_range_id is not None:
            torch_npu.npu.mstx.range_end(mstx_state.save_range_id)
            mstx_state.save_range_id = None
        return out

    return save_wrapper


def _apply_mstx_patch():
    DataLoader.__iter__ = _custom_dataloader_iter
    torch.serialization.save = _custom_save(original_save)