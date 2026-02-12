import os
import sys
import functools
import importlib

from typing import (
    Callable,
    List,
    Any,
    Dict
)

import torch
from time import time
from concurrent.futures import Future
from torch._inductor.utils import developer_warning

from torch._inductor.async_compile import (
    AsyncCompile,
    _compile_start,
    SubprocPool,
    get_compile_threads,
    _pool_set,
    log
)

from torch._dynamo.device_interface import get_interface_for_device

from torch._inductor.codecache import (
    config,
    Union,
    CodeCacheFuture,
    ModuleType,
    )

from .npu.mlir_compiler import NpuMlirCompiler
from .npu.codegen.akg import AkgCompiler
from . import config as anir_config
from .npu.utils import logger

class CompiledKernel:
    def __init__(self, kernel_call):
        self.kernel_call = kernel_call

    def run(self, *args, **kwargs):
        return self.kernel_call(*args, **kwargs)

def codegen_subgraph_dump(inds, shapes, strides, dtypes, inds2):
    codes = ["args = [arg.cuda() if isinstance(arg, torch.Tensor) else arg for arg in args]"]
    codes.append(f'new_args = [None] * {len(inds) + len(inds2)}')
    for ind, shape, stride, dtype in zip(inds, shapes, strides, dtypes):
        codes.append(f'new_args[{ind}] = rand_strided({shape}, {stride}, device="cuda:0", dtype={dtype})')
    codes.append(f'indices = {inds2}')
    codes.append(f'for i, ind in enumerate(indices):')
    codes.append(f'    new_args[ind] = args[i]')
    codes.append(f'args = new_args')
    return '\n'.join(codes)

    
def _worker_compile(
    kernel, cc: int, device: torch.device, logger_level=None, extra_env=None
) -> None:
    device_info = (device, device.index)
    try:
        kernel.get_best_kernel()
    except:
        kernel.precompile(device_info=device_info, logger_level=logger_level)

def _load_kernel(
        kernel_name: str, 
        source_code: str, 
        no_more_compile=False, 
        suppress_error=False, 
        kernel_meta=None,
        extra_env=None) -> ModuleType:
    device_str = kernel_meta.get('device_str')
    device_interface = get_interface_for_device(device_str)
    device = torch.device(device_str, device_interface.current_device())
    device_info = (device, device.index)
    kernel = NpuMlirCompiler(kernel_name, no_more_compile=no_more_compile, kernel_meta=kernel_meta)
    kernel.init(module=source_code, extra_env=extra_env)
    try:
        kernel.get_best_kernel()
    except:
        kernel.precompile(device_info=device_info, suppress_error=suppress_error)
    return kernel

def _load_fx_graph(kernel_name: str, source_code=None, extra_env=None, kernel_meta=None, autotune=True) -> ModuleType:
    kernel = NpuMlirCompiler(kernel_name, kernel_meta=kernel_meta, autotune=autotune)
    if source_code is not None:
        kernel.init(module=source_code, extra_env=extra_env)
    kernel.register_fx_fallback(kernel_meta)
    os.makedirs(os.path.join(kernel_meta.get('traced_graph_cache'), str(kernel_meta.get('device_index')), kernel_meta.get('traced_graph_hash'), 'keep'), exist_ok=True)
    return kernel

class MulitprocessCompileFuture(CodeCacheFuture):
    kernel: ModuleType

    def __init__(
        self,
        kernel_name: str,
        source_code: str,
        futures: List[Future],
        kernel_meta,
        extra_env,
    ) -> None:
        self.kernel_name = kernel_name
        self.source_code = source_code
        self.futures = futures
        self.kernel_meta = kernel_meta
        self.extra_env = extra_env

    # @dynamo_utils.dynamo_timed
    def result(self) -> ModuleType:
        t0 = time()
        if hasattr(self, "kernel"):
            return self.kernel
        errors = []
        for future in self.futures:
            try:
                future.result()
            except Exception as e:
                logger.warning(f"Error detected when multiprocess compile, error message: {e}")
                errors.append(e)

        if len(errors) < len(self.futures):
            kernel = self.kernel = _load_kernel(self.kernel_name, self.source_code, 
                                                no_more_compile=True, suppress_error=True,
                                                kernel_meta=self.kernel_meta, extra_env=self.extra_env)
        elif self.kernel_meta.get('num_outputs', 0): # All compiles fail and auto fallback
            print("==========================Kernel compiled failed!=======================================")
            print(f'kernel name: {self.kernel_name}')
            print(f'{self.source_code}')
            print("========================================================================================")
            kernel = self.kernel = _load_fx_graph(
                self.kernel_name, source_code=self.source_code, extra_env=self.extra_env, kernel_meta=self.kernel_meta)
        else:
            raise errors[0]

        latency = time() - t0
        if latency > 50:
            developer_warning(
                f"Detected long compilation time of {latency} seconds for kernel name {self.kernel_name}"
            )
            developer_warning(self.source_code)
        del self.kernel_name, self.source_code, self.futures
        return kernel


class NPUTritonFuture(CodeCacheFuture):
    kernel: ModuleType

    def __init__(
        self,
        kernel_name: str,
        source_code: str,
        future,
        kernel_meta,
        extra_env
    ) -> None:
        self.kernel_name = kernel_name
        self.source_code = source_code
        self.future = future
        self.kernel_meta = kernel_meta
        self.extra_env = extra_env

    # @dynamo_utils.dynamo_timed
    def result(self) -> ModuleType:
        t0 = time()
        if hasattr(self, "kernel"):
            return self.kernel
        # If the worker failed this will throw an exception.
        if self.kernel_meta.get('num_outputs'):
            try:
                self.future.result()
                kernel = self.kernel = _load_kernel(self.kernel_name, self.source_code, no_more_compile=True, kernel_meta=self.kernel_meta, extra_env=self.extra_env)
            except Exception as e:
                kernel = self.kernel = _load_fx_graph(
                    self.kernel_name, source_code=self.source_code, extra_env=self.extra_env, kernel_meta=self.kernel_meta)
        else:
            self.future.result()
            kernel = self.kernel = _load_kernel(self.kernel_name, self.source_code, no_more_compile=True, kernel_meta=self.kernel_meta, extra_env=self.extra_env)
        latency = time() - t0
        if latency > 50:
            developer_warning(
                f"Detected long compilation time of {latency} seconds for kernel name {self.kernel_name}"
            )
            developer_warning(self.source_code)
        del self.kernel_name, self.source_code, self.future
        return kernel


class CustomAsyncCompile(AsyncCompile):
    @staticmethod
    @functools.lru_cache(1)
    def process_pool() -> SubprocPool:
        assert get_compile_threads() > 1
        # Wrapper around ProcessPoolExecutor forks in a new process we control
        log.info("Creating subprocess pool with %d workers", get_compile_threads())
        pool = SubprocPool(get_compile_threads())

        # Set an attribute we can check to see if the pool is ready.
        pool.ready_future = pool.submit(AsyncCompile._get_ready)  # type: ignore[attr-defined]
        _pool_set.add(pool)
        return pool
    
    def mlir(
        self, kernel_name: str, source_code: str, device_str: str = "npu"
    ) -> Union[NPUTritonFuture, ModuleType]:
        if 'PY_DIR_PATH' in os.environ:
            raise RuntimeError('Stop early.')
        _compile_start()

        if config.compile_threads > 1:
            device_interface = get_interface_for_device(device_str)
            device = torch.device(device_str, device_interface.current_device())
            cc = device_interface.get_compute_capability(device)
            future = self.process_pool().submit(
                _worker_compile, kernel_name, source_code, cc, device, logger_level=logger.level
            )
            return NPUTritonFuture(kernel_name, source_code, future)
        else:
            return _load_kernel(kernel_name, source_code)
        
    def mlir_auto_fallback(
        self, kernel_name: str, source_code: str, kernel_meta: Dict[str, Any]) -> Callable:
        _compile_start()

        device_interface = get_interface_for_device(kernel_meta.get('device_str'))
        device = torch.device(kernel_meta.get('device_str'), device_interface.current_device())
        cc = device_interface.get_compute_capability(device)
        env_vars = ["TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"]
        extra_env = {v: os.environ[v] for v in env_vars if v in os.environ}

        if config.compile_threads > 1:
            if anir_config.multiprocess_compile:
                device_info = (device, device.index)
                kernel = NpuMlirCompiler(kernel_name, multiprocess_compile=True, kernel_meta=kernel_meta)
                kernel.init(module=source_code, extra_env=extra_env)
                try:
                    kernel.get_best_kernel()
                    return kernel
                except:
                    compile_args = kernel.get_autotune_config()
                    futures = []
                    for cargs in compile_args:
                        future = self.process_pool().submit(
                            kernel.compile_mlir, device_info, cargs, logger.level
                        )
                        futures.append(future)
                    return MulitprocessCompileFuture(kernel_name, source_code, futures, kernel_meta, extra_env)
            else:
                kernel = NpuMlirCompiler(kernel_name, multiprocess_compile=True, kernel_meta=kernel_meta)
                kernel.init(module=source_code, extra_env=extra_env)
                try:
                    kernel.get_best_kernel()
                    return kernel
                except:
                    future = self.process_pool().submit(
                        _worker_compile, kernel, cc, device, logger_level=logger.level, extra_env=extra_env
                    )
                    return NPUTritonFuture(kernel_name, source_code, future, kernel_meta, extra_env)
        else:
            kernel = _load_kernel(kernel_name, source_code, suppress_error=anir_config.autotune, kernel_meta=kernel_meta, extra_env=extra_env)
            if len(kernel.launchers) == 0:
                logger.info(f"fallback to fx graph call")
                return _load_fx_graph(kernel_name, source_code=source_code, extra_env=extra_env, kernel_meta=kernel_meta)
            return kernel

    def akg_auto_fallback(
        self, kernel_name: str, source_code: str, kernel_meta: Dict[str, Any]) -> Callable:
        _compile_start()
        kernel = AkgCompiler(kernel_meta=kernel_meta)
        kernel.compile(source_code)
        return kernel

    def import_fx(
        self, module_name: str, kernel_meta: Dict[str, Any]) -> Callable:

        device_interface = get_interface_for_device(kernel_meta.get('device_str'))
        device = torch.device(kernel_meta.get('device_str'), device_interface.current_device())
        cc = device_interface.get_compute_capability(device)
        env_vars = ["TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"]
        extra_env = {v: os.environ[v] for v in env_vars if v in os.environ}

        _compile_start()
        return _load_fx_graph(module_name, kernel_meta=kernel_meta, autotune=False)