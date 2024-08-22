import multiprocessing

import torch
import torch_npu


def _worker(i: int) -> None:
    torch_npu.npu.set_device(0)


def set_device():
    torch_npu.npu.set_device(0)
    multiprocessing.set_start_method("spawn", force=True)
    jobs = [multiprocessing.Process(target=_worker, args=(i,)) for i in range(70)]

    for p in jobs:
        p.start()

    for p in jobs:
        p.join()


set_device()
