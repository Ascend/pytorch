import os
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'mlir'
import sys
import functools
from typing import Callable, Dict, Any, Union, List, Tuple, Iterator
from itertools import count
import importlib
import tempfile
import subprocess
import shutil

import torch
import torch_npu
from torch._inductor.compile_fx import clone_preserve_strides
from torch._inductor.runtime.cache_dir_utils import triton_cache_dir

from .utils import (
    _build_npu_ext,
    replace_placeholders,
    do_bench,
    logger,
)
from .. import config as anir_config
from ..cache import get_cache_manager
from .._C import load_kernel_binary
from .codegen.cpp_wrapper import cpp_launcher


reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
global_cache = set()
_dump_id_iter: Iterator[int] = count()

class NpuMlirCompiler:
    def __init__(self, 
            kernel_name: str = '', 
            multiprocess_compile=False, 
            no_more_compile=False,
            kernel_meta=None,
            autotune=True):
        self.function = None
        self.mode = None
        self.launch = None
        self.dynamic = kernel_meta.get('dynamic')
        self.mutated_indices = kernel_meta.get('mutated_indices')
        self.kernel_hash = kernel_meta.get('kernel_hash')
        self.signature = kernel_meta.get('signature')
        self.ranks = kernel_meta.get('ranks')
        self.num_outputs = kernel_meta.get('num_outputs')
        self.num_call_functions = kernel_meta.get('num_call_functions')
        self.device_index = kernel_meta.get('device_index', 0)
        self.traced_graph_hash = kernel_meta.get('traced_graph_hash', 0)
        self.kernel_meta = kernel_meta
        if self.dynamic:
            self.get_host_func_and_tiling_size = None
        self.kernel_name = kernel_name
        self.launchers = []
        self.kernel_paths = []
        self.is_fallback_kernels = []
        self.multiprocess_compile = multiprocess_compile
        self.no_more_compile = no_more_compile
        self.mlir_processed = False
        self.fx_graph_launcher = None
        self.mlir_text = None
        self.non_contiguous_inputs = None
        self.non_contiguous_outputs = None
        self.autotuned = False
        self.autotune = autotune

    def init(self, module, extra_env):
        os.environ.update(extra_env)
        self.mlir_text = module
        if os.getenv("TRITON_CACHE_DIR") is None:
            os.environ["TRITON_CACHE_DIR"] = triton_cache_dir(
                self.kernel_meta.get("device_index", 0)
            )
        self.cache = get_cache_manager(self.kernel_hash)
        self.prepare_launch(need_pickle=self.multiprocess_compile)
        self.get_named_op_path()

    def register_fx_fallback(self, kernel_meta):
        def fx_graph_call(module: torch.nn.Module, num_outputs):
            def module_call(*args, **kwargs):
                actual_args = args[:-num_outputs]
                actual_outputs = module.forward(*actual_args)
                for out1, out2 in zip(actual_outputs, args[-num_outputs:]):
                    out2.data = out1.data
            return module_call
        num_outputs = kernel_meta.get('num_outputs', 0)
        traced_graph_hash = kernel_meta.get('traced_graph_hash')
        traced_graph_cache = os.path.join(os.getenv("TORCHINDUCTOR_CACHE_DIR"), kernel_meta.get('traced_graph_cache'))
        device_index = kernel_meta.get('device_index')
        dump_path = os.path.join(traced_graph_cache, str(device_index), traced_graph_hash)
        sys.path.append(dump_path)
        module = importlib.import_module(traced_graph_hash)
        sys.path.remove(dump_path)
        Model = getattr(module, traced_graph_hash)
        if Model is None:
            raise RuntimeError('Cannot find valid graph module!')
        model = Model()
        module_call = fx_graph_call(model, num_outputs)
        self.register_launcher(module_call, kernel_path=self.kernel_name + "_fx_fallback", is_fallback_kernel=True)

    def bisheng_compile(self,
                        input_path: str,
                        output_path: str,
                        auto_db=True,
                        ops_reorder=False,
                        tiling_size=None,
                        extra_command=None):
        bisheng_install_path = os.getenv('BISHENG_INSTALL_PATH', '')
        bisheng_ir_compile_path = os.path.join(bisheng_install_path, "bishengir-compile")
        command = [
            bisheng_ir_compile_path,
            "-enable-hfusion-compile=true",
            "--enable-bin-relocation=0",
            f"-block-dim={anir_config.block_dim}",
        ]
        if auto_db:
            command.append("--enable-auto-multi-buffer=true")
        else:
            command.append("--enable-auto-multi-buffer=false")

        if ops_reorder:
            command.append("--enable-ops-reorder=true")
        else:
            command.append("--enable-ops-reorder=false")

        if tiling_size is not None:
            command.append(f"--hfusion-max-buffer-count-tuning={tiling_size}")

        if anir_config.autotune:
            command.append("-enable-tuning-mode=true")

        if self.dynamic:
            command.append("--enable-static-bare-ptr=false")
            command.append("--enable-symbol-analysis=true")

        if isinstance(extra_command, list) and extra_command:
            command += extra_command
        command += [
            input_path,
            "-o", output_path
        ]
        logger.info(f"Start to compile, command is: [{' '.join(command)}]")
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)
            logger.info(f"[bisheng-compile success]")
        except subprocess.CalledProcessError as e:
            logger.info(f"[bisheng-compile failed]")
            logger.warning(f"Compile error msg: {e.stderr.decode('utf-8')}")
            raise e
        
    def prepare_launch(self, need_pickle=False):
        def get_launch_mod(so_path):
            spec = importlib.util.spec_from_file_location("__launcher", so_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        so_name = f"{self.kernel_name}.so"
        cache_so_path = self.cache.get_file(so_name)
        if cache_so_path is None or (anir_config.always_compile and cache_so_path not in global_cache):
            with tempfile.TemporaryDirectory() as tmpdir:
                c_wrapper_path = os.path.join(tmpdir, f"{self.kernel_name}_launch.cpp")
                cache_mlir_path = self.cache.put(cpp_launcher(self.signature, self.kernel_name, self.ranks, dynamic=self.dynamic), f"{self.kernel_name}_launch.cpp")
                global_cache.add(cache_mlir_path)
                with open(c_wrapper_path, 'w') as c_wrapper_file:
                    c_wrapper_file.write(cpp_launcher(self.signature, self.kernel_name, self.ranks, dynamic=self.dynamic))
                if self.dynamic:
                    with open(c_wrapper_path, "rb") as f:
                        cache_c_wrapper_path = self.cache.put(f.read(), f"{self.kernel_name}_launch.cpp", binary=True)
                    global_cache.add(cache_c_wrapper_path)
                so_path = _build_npu_ext(self.kernel_name, c_wrapper_path, tmpdir)
                with open(so_path, "rb") as f:
                    cache_so_path = self.cache.put(f.read(), so_name, binary=True)
                global_cache.add(cache_so_path)
        if not need_pickle:
            mod = get_launch_mod(cache_so_path)
            self.launch = getattr(mod, "launch")
            if self.dynamic:
                self.get_host_func_and_tiling_size = getattr(mod, "get_host_func_and_tiling_size")

    def get_named_op_path(self):
        named_op_name = f"{self.kernel_name}_named_op.mlir"
        cache_mlir_path = self.cache.get_file(named_op_name)
        if cache_mlir_path is None or (anir_config.always_compile and cache_mlir_path not in global_cache):
            #if anir_config.cache_named_op:
            cache_mlir_path = self.cache.put(self.mlir_text, named_op_name)
            global_cache.add(cache_mlir_path)
        if anir_config.fx_subgraph_dump_path:
            shutil.copy(cache_mlir_path, os.path.join(anir_config.fx_subgraph_dump_path, \
                                                      str(self.device_index), self.kernel_name))
        return cache_mlir_path
    
    def get_launch_dynamic(self, function, tiling_func, tiling_size):
        block_dim = anir_config.block_dim
        arg_tiling_device = torch.empty((tiling_size // 8), device='npu', dtype=torch.int64)
        arg_tiling_host = torch.empty((tiling_size // 8), dtype=torch.int64)
        def kernel_call(*args, stream=None):
            self.launch(block_dim, stream, function, tiling_func, tiling_size, arg_tiling_host, arg_tiling_device, None, None, None, *args)
        return kernel_call
    
    def get_launch(self, function):
        block_dim = anir_config.block_dim
        def kernel_call(*args, function, stream=None):
            self.launch(block_dim, stream, function, None, None, None, *args)

        return functools.partial(kernel_call, function=function)
    
    def get_launch_func(self, cache_kernel_path):
        if self.dynamic:
            function, tiling_func, tiling_size = self.get_host_func_and_tiling_size(self.kernel_name, 
                                                                                    self.kernel_name + '_tiling_function', 
                                                                                    self.kernel_name + '_get_tiling_struct_size_function', 
                                                                                    cache_kernel_path)
            return self.get_launch_dynamic(function, tiling_func, tiling_size)
        else:
            function = load_kernel_binary(self.kernel_name, cache_kernel_path)
            return self.get_launch(function)
    
    def register_launcher(self, 
                          launcher, 
                          kernel_path=None, 
                          num_outputs=None, 
                          disable_dump=False, 
                          auto_fallback=False,
                          is_fallback_kernel=False):
        if num_outputs:
            self.num_outputs = num_outputs
        self.launchers.append(launcher)
        self.kernel_paths.append(kernel_path)
        self.is_fallback_kernels.append(is_fallback_kernel)
        self.fx_graph_launcher = launcher
        if kernel_path.endswith('_fx_fallback'):
            if auto_fallback:
                if anir_config.fallback_warning:
                    print(f"This kernel {self.kernel_name} has been fallback to the eager fx graph mode, ", \
                    "which will lead to a significant decrease in performance.", flush=True)
                if anir_config.fallback_dump and not disable_dump:
                    self.fx_subgraph_dump('fallback')
        logger.info(f"register launcher {launcher} {kernel_path} success")

    def compile_mlir(self, 
                     device_info: Tuple[Any],
                     compile_args: List[Any],
                     logger_level = None) -> Callable[..., None]:
        if logger_level is not None:
            # re-init logger level in subprocess
            logger.setLevel(logger_level)
        named_op_mlir_path = self.get_named_op_path()

        kernel_name = self.kernel_name

        tiling_size, ops_reorder, auto_db = compile_args
        tiling_str = f"_{tiling_size}_{ops_reorder}_{auto_db}"
        tiling_kernel_name = kernel_name + tiling_str
        if self.dynamic:
            cache_kernel_path = self.cache.get_file(f"lib{tiling_kernel_name}.so")
        else:
            cache_kernel_path = self.cache.get_file(f"{tiling_kernel_name}.o")

        logger.info("Start to get cached kernel. Tiling info: " +
                    f"tiling_size {tiling_size} ops_reorder {ops_reorder} auto_db {auto_db}")
        
        if cache_kernel_path is None and self.no_more_compile:
            raise RuntimeError("Skip compile.")

        if cache_kernel_path is None or (anir_config.always_compile and cache_kernel_path not in global_cache):
            logger.info("No cached kernel. Start to exec compile.")
            with tempfile.TemporaryDirectory() as tmpdir:
                kernel_path = os.path.join(tmpdir, tiling_kernel_name)
                self.bisheng_compile(named_op_mlir_path, kernel_path, tiling_size=tiling_size,
                                    ops_reorder=ops_reorder, auto_db=auto_db,
                                    extra_command=anir_config.extra_command)
                
                
                if self.dynamic:
                    kernel_path = os.path.join(tmpdir, f"lib{tiling_kernel_name}.so")
                    with open(kernel_path, "rb") as f:
                        cache_kernel_path =  self.cache.put(f.read(), f"lib{tiling_kernel_name}.so", binary=True)
                    global_cache.add(cache_kernel_path)
                else:
                    kernel_path =  os.path.join(tmpdir, tiling_kernel_name + '.o')
                    with open(kernel_path, "rb") as f:
                        cache_kernel_path =  self.cache.put(f.read(), f"{tiling_kernel_name}.o", binary=True)
                    global_cache.add(cache_kernel_path)

        logger.info("Get kernel success.")
        if not self.multiprocess_compile:
            logger.info(f"Start to register kernel, path '{cache_kernel_path}' func '{self.kernel_name}'")
            launch_func = self.get_launch_func(cache_kernel_path)
            self.register_launcher(launch_func, cache_kernel_path)

        if anir_config.fx_subgraph_dump_path:
            kernel_dump_path = os.path.join(anir_config.fx_subgraph_dump_path, \
                                            str(self.device_index), self.kernel_name, 'kernel_dump')
            os.makedirs(kernel_dump_path, exist_ok=True)
            shutil.copy(cache_kernel_path, kernel_dump_path)

    def replace_kernel_by_path(self, kernel_path: str):
        self.launchers.clear()
        self.kernel_paths.clear()
        self.is_fallback_kernels.clear()
        logger.info(f"Start to replace kernel by specific path, path '{kernel_path}' func '{self.kernel_name}'")
        launch_func = self.get_launch_func(kernel_path)
        self.register_launcher(launch_func, kernel_path)

    def get_best_kernel(self):
        def get_launch_mod(so_path):
            spec = importlib.util.spec_from_file_location("__launcher", so_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        best_kernel = self.cache.get_file('best_kernel')
        if best_kernel is None:
            raise RuntimeError("can not find best kernel")
        with open(best_kernel, 'r') as f:
            kernel_path = self.cache.get_file(f.read())
        if not kernel_path.endswith(('.so', '.o')):
            self.register_fx_fallback(self.kernel_meta)
            return
        so_path = self.cache.get_file(f'{self.kernel_name}.so')
        if kernel_path is None:
            return RuntimeError()
        mod = get_launch_mod(so_path)
        self.launch = getattr(mod, "launch")
        if self.dynamic:
            self.get_host_func_and_tiling_size = getattr(mod, "get_host_func_and_tiling_size")
        
        launch_func = self.get_launch_func(kernel_path)
        self.register_launcher(launch_func, kernel_path)
        return True

    def get_autotune_config(self):
        def get_tiling_range():
            return [i for i in range(-10, 20, 2)]
        compile_args = []
        for ops_reorder in [True, False]:
            for auto_db in [True, False]:
                for tiling_size in get_tiling_range():
                    compile_args.append((tiling_size, ops_reorder, auto_db))
        return compile_args

    def precompile(self, 
                    device_info: Tuple[Any],
                    suppress_error=False,
                    logger_level=None):

        if anir_config.autotune:
            compile_args = self.get_autotune_config()
        else:
            compile_args = [(None, True, True)]
        for cargs in compile_args:
            try:
                self.compile_mlir(device_info, cargs, logger_level=logger_level)
            except Exception as e:
                if suppress_error:
                    logger.warning(f"compile args {cargs} fail, err msg: {e}")
                else:
                    raise e

    def bench(self, idx, launcher, *args, **kwargs):
        if anir_config.runtime_error_dump:
            self.data_dump_fake(*args)
        cloned_args = args
        def kernel_call():
            launcher(*cloned_args, **kwargs)
        try:
            return do_bench(kernel_call, warmup=1, rep=5, fast_flush=True)
        except Exception as e:
            print(f"RUNTIME ERROR: eval kernel fail, kernel path: {self.kernel_paths[idx]}, ",
                f"try to add {self.kernel_paths[idx]} to anir_config.force_fallback_kernel_paths", flush=True)
            print(e, flush=True)
            if anir_config.runtime_error_dump:
                self.fx_subgraph_dump('runtime_error')
            exit(0)

    def clone_args(self, *args, **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        # [Note: clone mutated buffers]
        # clone inplace buffers to avoid autotune contaminating them if
        # the kernel does in-place stores. avoid cloning other buffers because
        # it leads to increase memory use
        cloned_args = []
        for i, arg in enumerate(args):
            if i in self.mutated_indices:
                assert isinstance(arg, torch.Tensor)
                cloned_args.append(clone_preserve_strides(arg))
            else:
                cloned_args.append(arg)

        return cloned_args

    def accuracy_pass(self, fx_outputs, *args):
        num_inputs = len(args) - self.num_outputs
        for _, (actual, expected) in enumerate(zip(args[num_inputs:], fx_outputs)):
            if actual.dtype != expected.dtype:
                expected = expected.to(actual.dtype)
            acc_comp_tol = anir_config.acc_comp_tol.get(actual.dtype, anir_config.acc_comp_tol['default'])
            rtol = acc_comp_tol['rtol']
            atol = acc_comp_tol['atol']
            matches = torch.allclose(
                actual, expected, rtol=rtol, atol=atol, equal_nan=True
            )
            if not matches:
                return False

        return True

    def benchmark_all_configs(self, *args, **kwargs):
        timings = []
        args_new = ()
        args = list(args)

        launcher_fx = self.launchers[-1]
        fx_outputs = [clone_preserve_strides(arg).to(torch.float32) if arg.dtype == torch.bfloat16 \
                      else clone_preserve_strides(arg) for arg in args[-self.num_outputs:]]
        fx_inputs = [clone_preserve_strides(arg) if isinstance(arg, torch.Tensor) else arg for arg in args[:-self.num_outputs]]
        fx_inputs = [inp.float() if isinstance(inp, torch.Tensor) and inp.dtype == torch.bfloat16 else inp for inp in fx_inputs]
        
        fx_args = fx_inputs + fx_outputs
        launcher_fx(*fx_args, **kwargs)

        if self.dynamic:
            for idx, arg in enumerate(args):
                if not torch.is_tensor(arg):
                    args_new = args_new + (arg, )
                    continue
                if idx in self.mutated_indices:
                    cloned_arg = clone_preserve_strides(arg)
                    args[idx] = cloned_arg
                    args_new = args_new + (cloned_arg, cloned_arg, 0) + arg.size() + arg.stride()
                else:
                    args_new = args_new + (arg, arg, 0) + arg.size() + arg.stride()
        else:
            for idx, arg in enumerate(args):
                if torch.is_tensor(arg) and idx in self.mutated_indices:
                    cloned_arg = clone_preserve_strides(arg)
                    args_new = args_new + (cloned_arg, )
                    args[idx] = cloned_arg
                else:
                    args_new = args_new + (arg, )

        for idx, launcher in enumerate(self.launchers):
            if self.dynamic and not self.is_fallback_kernels[idx]:
                transformed_args = args_new
            else:
                transformed_args = args
            if self.kernel_name in anir_config.force_fallback_kernel_names and not self.kernel_paths[idx].endswith('_fx_fallback'):
                continue
            if self.kernel_paths[idx] in anir_config.force_fallback_kernel_paths:
                print(f"Skip kernel: {self.kernel_paths[idx]}", flush=True)
                continue
            if self.is_fallback_kernels[idx] and not anir_config.autotune_fx_fallback:
                continue
            try:
                logger.info(f"start to eval kernel {self.kernel_paths[idx]}")
                times = self.bench(idx, launcher, *transformed_args, **kwargs)
                if self.accuracy_pass(fx_outputs, *transformed_args):
                    timings.append([times, idx])
                logger.info(f"eval over")
            except Exception as e:
                print(e)
                continue
        if not timings:
            timings.append([float(1.0), len(self.launchers) - 1])
        return timings
    
    def autotune_to_one_config(self, *args, **kwargs):
        if any([isinstance(arg, torch.Tensor) and not arg.is_contiguous() for arg in args]):
            print(f'Non contiguous args exists! Kernel name is {self.kernel_name}')
        timings = self.benchmark_all_configs(*args, **kwargs)
        timings.sort()
        logger.info(f"autotune over, timings: {timings}")
        if timings[0][0] > 99999:
            raise RuntimeError("All config exec failed.")
        idx = timings[0][1]
        logger.info(f"autotune benchmark over, using kernel {self.kernel_paths[idx]}")
        self.kernel_paths = [self.kernel_paths[idx]]
        self.launchers = [self.launchers[idx]]
        self.is_fallback_kernels = [self.is_fallback_kernels[idx]]
        if self.is_fallback_kernels[0]:
            self.cache.put(self.traced_graph_hash, "best_kernel", binary=False)
        else:
            self.cache.put(self.kernel_paths[0].split('/')[-1], "best_kernel", binary=False)

    def data_dump(self, *args, dump_path=None):
        if not dump_path:
            dump_path = os.path.join(anir_config.fx_subgraph_dump_path, str(self.device_index), self.kernel_name)
        data_dump_path = os.path.join(dump_path, 'data.pth')
        args_cpu = [arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args]
        torch.save(args_cpu, data_dump_path)

    def data_dump_fake(self, *args, dump_path=None):
        if not dump_path:
            dump_path = os.path.join(anir_config.fx_subgraph_dump_path, str(self.device_index), self.kernel_name)
        runable_py_path = os.path.join(dump_path, f'runnable_{self.kernel_name}.py')
        fake_inputs = [f'rand_strided({arg.shape}, {arg.stride()}, device="{arg.device.type}", dtype={arg.dtype})' \
                       if isinstance(arg, torch.Tensor) else str(arg) for arg in args[:-self.num_outputs]]
        fake_outputs = [f'empty_strided({arg.shape}, {arg.stride()}, device="{arg.device.type}", dtype={arg.dtype})' \
                        if isinstance(arg, torch.Tensor) else str(arg) for arg in args[-self.num_outputs:]]
        replacements = {"FAKE_ARGS_PLACEHOLDER": f"args = [{', '.join(fake_inputs + fake_outputs)}]"}
        replace_placeholders(runable_py_path, replacements)

    def fx_subgraph_dump(self, suffix):
        subgraph_dump_path = os.path.join(anir_config.fx_subgraph_dump_path, str(self.device_index), self.kernel_name)
        failed_fx_subgraph_dump_path = anir_config.fx_subgraph_dump_path + f'_{suffix}'
        failed_subgraph_dump_path = os.path.join(failed_fx_subgraph_dump_path, str(self.device_index), f'{next(_dump_id_iter)}_' + self.kernel_name)
        if os.path.exists(failed_subgraph_dump_path):
            shutil.rmtree(failed_subgraph_dump_path)
        shutil.copytree(subgraph_dump_path, failed_subgraph_dump_path)
        return failed_subgraph_dump_path
        
    def acc_compare_and_dump(self, *args, **kwargs):
        from torch.testing._comparison import _make_mismatch_msg
        self.register_fx_fallback(self.kernel_meta)
        launcher_fx = self.launchers[1]
        launcher = self.launchers[0]

        fx_outputs = [clone_preserve_strides(arg).to(torch.float32) if arg.dtype == torch.bfloat16 \
                      else clone_preserve_strides(arg) for arg in args[-self.num_outputs:]]
        fx_inputs = [clone_preserve_strides(arg) if isinstance(arg, torch.Tensor) else arg for arg in args[:-self.num_outputs]]
        fx_inputs = [inp.float() if isinstance(inp, torch.Tensor) and inp.dtype == torch.bfloat16 else inp for inp in fx_inputs]
        
        fx_args = fx_inputs + fx_outputs
        launcher_fx(*fx_args, **kwargs)

        if self.dynamic:
            args_new = ()
            for arg in args:
                if not torch.is_tensor(arg):
                    args_new = args_new + (arg, )
                    continue
                args_new = args_new + (arg, arg, 0) + arg.size() + arg.stride()
        else:
            args_new = args
        
        output = launcher(*args_new, **kwargs)

        has_acc_error = False
        num_inputs = len(args) - self.num_outputs
        for idx, (actual, expected) in enumerate(zip(args[num_inputs:], fx_outputs)):
            if actual.dtype != expected.dtype:
                expected = expected.to(actual.dtype)
            acc_comp_tol = anir_config.acc_comp_tol.get(actual.dtype, anir_config.acc_comp_tol['default'])
            rtol = acc_comp_tol['rtol']
            atol = acc_comp_tol['atol']
            matches = torch.isclose(
                actual, expected, rtol=rtol, atol=atol, equal_nan=True
            )
            if not matches.all():
                abs_diff = abs(actual - expected)
                rel_diff = abs_diff / abs(expected)
                rel_diff.masked_fill_(matches, 0)
                number_of_elements = matches.numel()
                total_mismatches = number_of_elements - int(torch.sum(matches))
                extra = (
                    f"Mismatched elements: {total_mismatches} / {number_of_elements} "
                    f"({total_mismatches / number_of_elements:.1%})"
                )
                msg = _make_mismatch_msg(
                    default_identifier="Tensor-likes",
                    identifier=None,
                    extra=extra,
                    abs_diff=abs_diff.max().item(),
                    abs_diff_idx=None,
                    atol=atol,
                    rel_diff=rel_diff.max().item(),
                    rel_diff_idx=None,
                    rtol=rtol,
                )
                print(f"Kernel Name: {self.kernel_name}\n{msg}", flush=True)
                has_acc_error = True
                args[idx + num_inputs].copy_(expected)
                del abs_diff
                del rel_diff
            del matches
            del expected
        
        if anir_config.fx_subgraph_dump_path:
            data = args
            if has_acc_error:
                data_dump_path = self.fx_subgraph_dump('acc_failed')
                self.data_dump_fake(*data, dump_path=data_dump_path)
        del fx_inputs
        torch.npu.synchronize()
        self.launchers = [self.launchers[0]]
        self.is_fallback_kernels = [self.is_fallback_kernels[0]]
        
        return output
    
    def mlir_dump(self, *args, **kwargs):
        self.data_dump(*args)
        launcher_fx = self.launchers[-1]
        fx_output = launcher_fx(*args, **kwargs)
        return fx_output

    def make_inputs_contiguous(self, args):
        args = list(args)
        for idx in self.non_contiguous_indices['inputs']:
            args[idx] = args[idx].contiguous()
        return tuple(args)
        
    def run(self, *args, **kwargs):
        args = list(args)
        
        if self.non_contiguous_inputs is None:
            self.non_contiguous_inputs = []
            if self.num_call_functions > 0:
                for idx, arg in enumerate(args[:-self.num_outputs]):
                    if isinstance(arg, torch.Tensor) and not arg.is_contiguous():
                        args[idx] = args[idx].contiguous()
                        self.non_contiguous_inputs.append(idx)
        else:
            for idx in self.non_contiguous_inputs:
                args[idx] = args[idx].contiguous()

        contiguous_outputs = []
        
        if self.non_contiguous_outputs is None:
            self.non_contiguous_outputs = []
            original_outputs = []
            num_inputs = len(args) - self.num_outputs
            for idx, arg in enumerate(args[num_inputs:]):
                if isinstance(arg, torch.Tensor) and not arg.is_contiguous():
                    contiguous_output = torch.empty(
                        arg.shape, 
                        dtype=arg.dtype, 
                        device=arg.device)
                    arg_idx = idx - self.num_outputs
                    original_outputs.append(arg)
                    args[arg_idx] = contiguous_output
                    self.non_contiguous_outputs.append(arg_idx)
                    contiguous_outputs.append(contiguous_output)
        else:
            original_outputs = []
            for idx in self.non_contiguous_outputs:
                contiguous_output = torch.empty(
                    args[idx].shape, 
                    dtype=args[idx].dtype, 
                    device=args[idx].device)
                original_outputs.append(args[idx])
                args[idx] = contiguous_output
                contiguous_outputs.append(contiguous_output)
    
        if not self.autotuned:
            if self.autotune:
                self.register_fx_fallback(self.kernel_meta)
                self.autotune_to_one_config(*args, **kwargs)
            else:
                pass
            self.autotuned = True

        (launcher,) = self.launchers
        (is_fallback_kernel, ) = self.is_fallback_kernels

        if anir_config.fx_subgraph_dump_path and \
            anir_config.online_acc_comp and \
            not is_fallback_kernel:
            output = self.acc_compare_and_dump(*args, **kwargs)
            if self.non_contiguous_outputs:
                for i, idx in enumerate(self.non_contiguous_outputs):
                    original_outputs[i].copy_(args[idx])
            return output

        if self.dynamic and not is_fallback_kernel:
            args_new = ()
            for arg in args:
                if not torch.is_tensor(arg):
                    args_new = args_new + (arg, )
                    continue
                args_new = args_new + (arg, arg, 0) + arg.size() + arg.stride()
            args = args_new

        output = launcher(*args, **kwargs)

        if self.non_contiguous_outputs:
            for i, idx in enumerate(self.non_contiguous_outputs):
                original_outputs[i].copy_(args[idx])

        del contiguous_outputs
        return output