from __future__ import annotations

import contextlib
import copy
import ctypes
import dataclasses
import functools
import logging
import os
import queue
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from ctypes import CDLL, byref, c_size_t, c_void_p
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Optional, Sequence, Union)

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch import multiprocessing
from torch._dynamo.testing import rand_strided
from torch._inductor import config, ir
from torch._inductor.autotune_process import (
    BenchmarkRequest, NonzeroWorkspaceNotSupportedError, TensorMeta)
from torch._inductor.codecache import DLLWrapper
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.virtualized import V

from .codecache import CATLASSCodeCache


ASCEND_VISIBLE_DEVICES = "ASCEND_RT_VISIBLE_DEVICES"
EXIT_HANDLER_REGISTERED = False

log = logging.getLogger("torch._inductor")


def patch_tuning_process():
    from torch._inductor import autotune_process

    autotune_process.CUDA_VISIBLE_DEVICES = ASCEND_VISIBLE_DEVICES


def patch_tuning_process_pool():
    from torch._inductor.autotune_process import TuningProcessPool

    def get_device_list(self) -> Sequence[Optional[int]]:
        """
        Gather the list of devices to be used in the pool.
        """
        if not config.autotune_multi_device:
            # Don't use multiple devices
            return [None]

        count = torch.npu.device_count()

        # If the user specified the visible devices in the env, use those.
        if ASCEND_VISIBLE_DEVICES in os.environ:
            devices = [int(d) for d in os.environ[ASCEND_VISIBLE_DEVICES].split(",")]
            if len(devices) > count:
                raise ValueError(f"Specified visible devices exceed the number of total devices: {devices}")
            return devices

        return list(range(count))

    TuningProcessPool.get_device_list = get_device_list


class NPUDeviceBenchmarkMixin:
    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        device_idx_set = {
            tensor.device.index
            for tensor in [*input_tensors, output_tensor]
            if isinstance(tensor, torch.Tensor)
            and tensor.is_npu
            and tensor.device.index is not None
        }
        if len(device_idx_set) > 1:
            raise ValueError(f"Can not mix devices: {device_idx_set}")
        if len(device_idx_set) == 1:
            device_idx = next(iter(device_idx_set))
        else:
            device_idx = torch.npu.current_device()

        with torch.npu.device(device_idx):
            out = self._bench(fn)
            torch.npu.synchronize()  # shake out any NPU errors

        return out

    def _bench(
        self,
        fn,
        warmup=25,
        repeats=100,
    ) -> float:
        fn()
        torch.npu.synchronize()

        # Estimate the runtime of the function
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            fn()
        end_event.record()
        torch.npu.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5

        # compute number of warmup and repeat
        n_warmup = min(max(int(warmup / estimate_ms), 1), 250)
        n_repeat = min(max(int(repeats / estimate_ms), 1), 1000)

        # warm-up
        for _ in range(n_warmup):
            fn()
        # benchmark
        start_event.record()
        for _ in range(n_repeat):
            fn()
        end_event.record()
        torch.npu.synchronize()

        return start_event.elapsed_time(end_event) / n_repeat


class CATLASSBenchmarkRequest(NPUDeviceBenchmarkMixin, BenchmarkRequest):
    # Important: Instances of this class have to be serializable
    # across process boundaries. Do not put NPU Tensors in here!

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        extra_args: Iterable[Any],
        source_code: str,
        is_mix: bool = False,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code
        self.is_mix = is_mix
        self.workspace_size: int = 0
        self.workspace: Optional[torch.Tensor] = None
        self.DLL: Optional[DLLWrapper] = None
        self._workspace_size_updated = False
        self.hash_key: str = ""
        self.source_file: str = ""
        self.hash_key, self.source_file = CATLASSCodeCache.write(self.source_code, "so", self.is_mix)

    def benchmark(
        self,
        *input_tensors: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        # create args and out tensor
        if output_tensor is None:
            input_tensors = tuple(x.to_tensor() for x in self.input_tensor_meta)
            output_tensor = self.output_tensor_meta.to_tensor()

        try:
            fn = self.make_run_fn(*input_tensors, output_tensor=output_tensor)
        except NonzeroWorkspaceNotSupportedError:
            log.info("Skipping op due to nonzero workspace requirement")
            return float("inf")

        out = self.do_bench(fn, *input_tensors, output_tensor)
        return out

    def precompile(self):
        # Prepopulate CATLASSCodeCache
        # may happen in separate Threadpool
        log.debug("Precompiling %s", self)
        CATLASSCodeCache.compile(self.source_code, "so", is_mix=self.is_mix)
        log.debug("Done precompiling %s", self)

    def make_run_fn(
        self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor
    ) -> Callable[[], None]:
        self.ensure_dll_loaded()
        self.update_workspace_size()
        args = [
            c_void_p(tensor.data_ptr())
            for tensor in list(input_tensors) + [output_tensor]
        ]
        log.debug(
            "make_run_fn: self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        stream_ptr = c_void_p(torch.npu.current_stream().npu_stream)
        run_method = getattr(self.DLL, self.kernel_name)
        workspace_ptr = c_void_p(0)
        if self.workspace_size > 0:
            self.workspace = torch.zeros(
                (self.workspace_size + 7) // 8,
                dtype=torch.float64,
                device=output_tensor.device,
            )
            workspace_ptr = c_void_p(self.workspace.data_ptr())

        # Generate partial function.
        return functools.partial(
            run_method,
            *args,
            *self.extra_args,
            None,  # null workspace size ptr
            workspace_ptr,  # set workspace ptr,
            stream_ptr,
        )

    def update_workspace_size(self) -> None:
        if self._workspace_size_updated:
            return
        self.ensure_dll_loaded()
        unique_input_count = len({meta.name for meta in self.input_tensor_meta})
        args = [c_void_p(None) for _ in range(unique_input_count + 1)]
        stream_ptr = c_void_p(torch.npu.current_stream().npu_stream)

        run_method = getattr(self.DLL, self.kernel_name)
        # Retrieve workspace_size and initialize workspace.
        c_workspace_size = c_size_t()
        run_method(
            *args,  # input ptrs and output ptrs
            *self.extra_args,
            byref(
                c_workspace_size
            ),  # set workspace size ptr to retrieve workspace size
            None,  # null workspace ptr
            stream_ptr,
        )
        torch.npu.synchronize()  # shake out any NPU errors
        self.workspace_size = c_workspace_size.value
        log.debug(
            "update_workspace_size called: new workspace size=%d, self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",  # noqa: B950
            self.workspace_size,
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        self._workspace_size_updated = True

    def ensure_dll_loaded(self):
        if self.DLL is None:
            self.DLL, self.hash_key, self.source_file = CATLASSCodeCache.load(
                self.source_code, "so", self.is_mix
            )

    def cleanup_run_fn(self) -> None:
        if self.DLL is not None:
            self.DLL.close()
            self.DLL = None
        self.workspace = None

    def __str__(self) -> str:
        return f"{self.kernel_name=}, {self.source_file=}, {self.hash_key=}"


class FusedCATLASSBenchmarkRequest():
    def __init__(
        self,
        kernel_name,
        src_code,
        template_node,
        epilogue_nodes,
        extra_args,
    ):
        self.kernel_name = kernel_name
        self.src_code = src_code
        self.template_node = template_node
        self.epilogue_nodes = epilogue_nodes
        self.extra_args = extra_args
        self.bmreq = None

    def get_intput_outputs(self):
        kernel_inputs = copy.copy(self.template_node.node.inputs)
        kernel_outputs = self.template_node.node

        for epi_node in self.epilogue_nodes:
            for name in epi_node.node.get_read_names():
                if name == self.template_node.node.get_name():
                    continue
                inp_buf = None
                if name in V.graph.name_to_buffer:
                    inp_buf = V.graph.name_to_buffer[name]
                elif name in V.graph.graph_inputs:
                    inp_buf = V.graph.graph_inputs[name]
                if inp_buf in kernel_inputs:
                    continue
                assert inp_buf is not None
                kernel_inputs.append(inp_buf)
        return kernel_inputs, kernel_outputs

    def benchmark(self):
        kernel_inputs, kernel_outputs = self.get_intput_outputs()

        self.bmreq = CATLASSBenchmarkRequest(
            kernel_name=self.kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(kernel_inputs),
            output_tensor_meta=TensorMeta.from_irnodes(kernel_outputs),
            extra_args=self.extra_args,
            source_code=self.src_code,
            is_mix=self.template_node.node.is_mix,
        )
        example_inputs = ()
        res = self.bmreq.benchmark(*example_inputs)
        return res, None
