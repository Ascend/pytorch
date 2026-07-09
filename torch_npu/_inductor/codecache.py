import os
import contextlib

import torch
from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT
import torch_npu  # noqa: F401


@contextlib.contextmanager
def lock_context(key):
    from filelock import FileLock
    lock_dir = get_lock_dir()
    lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
    with lock:
        yield


def patch_get_cpp_wrapper_header():
    origin_get_cpp_wrapper_header = torch._inductor.codecache._get_cpp_wrapper_header

    def _get_cpp_wrapper_header_npu(device: str, aot_mode: bool = False) -> str:
        base_device = device.split(":", maxsplit=1)[0]
        if base_device == "npu":
            return (
                "torch_npu/csrc/inductor/"
                f"{'aoti_include' if aot_mode else 'cpp_wrapper'}/"
                f"{base_device}.h"
            )
        return origin_get_cpp_wrapper_header(device, aot_mode)

    torch._inductor.codecache._get_cpp_wrapper_header = _get_cpp_wrapper_header_npu
