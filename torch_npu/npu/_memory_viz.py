import sys
import os
import io
import subprocess
import json
from functools import lru_cache
from typing import Any
from itertools import groupby
import base64
import warnings
import yaml
from torch.cuda._memory_viz import format_flamegraph
import torch_npu

from torch_npu.utils._error_code import ErrCode, pta_error

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.realpath(torch_npu.__file__))

cache = lru_cache(None)

__all__ = ["segments", "memory"]


def _frame_fmt(f, full_filename=False):
    i = f['line']
    fname = f['filename']
    if not full_filename:
        fname = fname.split('/')[-1]
    func = f['name']
    return f'{fname}:{i}:{func}'


@cache
def _frame_filter(name, filename):
    omit_functions = [
        "unwind::unwind",
        "CapturedTraceback::gather",
        "gather_with_cpp",
        "_start",
        "__libc_start_main",
        "PyEval_",
        "PyObject_",
        "PyFunction_",
    ]
    omit_filenames = [
        "core/boxing",
        "/Register",
        "/Redispatch",
        "pythonrun.c",
        "Modules/main.c",
        "Objects/call.c",
        "Objects/methodobject.c",
        "pycore_ceval.h",
        "ceval.c",
        "cpython/abstract.h",
    ]
    for of in omit_functions:
        if of in name:
            return False
    for of in omit_filenames:
        if of in filename:
            return False
    return True


def _frames_fmt(frames, full_filename=False, reverse=False):
    if reverse:
        frames = reversed(frames)
    return [_frame_fmt(f, full_filename) for f in frames if _frame_filter(f['name'], f['filename'])]


def _block_extra_legacy(b):
    if 'history' in b:
        frames = b['history'][0].get('frames', [])
        real_size = b['history'][0]['real_size']
    else:
        real_size = b.get('requested_size', b['size'])
        frames = []
    return frames, real_size


def _block_extra(b):
    if 'frames' not in b:
        # old snapshot format made it more complicated to get frames/allocated size
        return _block_extra_legacy(b)
    return b['frames'], b['requested_size']


def _write_blocks(f, prefix, blocks):
    def frames_fragment(frames):
        if not frames:
            return "<non-python>"
        return ';'.join(_frames_fmt(frames, reverse=True))
    for b in blocks:
        if 'history' not in b:
            frames, accounted_for_size = _block_extra(b)
            f.write(f'{prefix};{b["state"]};{frames_fragment(frames)} {accounted_for_size}\n')
        else:
            accounted_for_size = 0
            for h in b['history']:
                sz = h['real_size']
                accounted_for_size += sz
                if 'frames' in h:
                    frames = h['frames']
                    f.write(f'{prefix};{b["state"]};{frames_fragment(frames)} {sz}\n')
                else:
                    f.write(f'{prefix};{b["state"]};<no-context> {sz}\n')
        gaps = b['size'] - accounted_for_size
        if gaps:
            f.write(f'{prefix};{b["state"]};<gaps> {gaps}\n')


def segments(snapshot, format_flamegraph_func=format_flamegraph):
    f = io.StringIO()
    for seg in snapshot['segments']:
        prefix = f'stream_{seg["stream"]};seg_{seg["address"]}'
        _write_blocks(f, prefix, seg['blocks'])
    return format_flamegraph_func(f.getvalue())


def memory(snapshot, format_flamegraph_func=format_flamegraph):
    f = io.StringIO()
    for seg in snapshot['segments']:
        prefix = f'stream_{seg["stream"]}'
        _write_blocks(f, prefix, seg['blocks'])
    return format_flamegraph_func(f.getvalue())
