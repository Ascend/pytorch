import os
import sys
import importlib
import shutil
from itertools import count
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator

import torch
from torch._inductor.compile_fx import clone_preserve_strides

from .. import config as anir_config
from .utils import replace_placeholders


_dump_id_iter: Iterator[int] = count()

class MetaCompiler:
    def __init__(
        self,
        kernel_name: str = "",
        multiprocess_compile: bool = False,
        no_more_compile: bool = False,
        kernel_meta: Optional[Dict[str, Any]] = None,
        autotune: bool = True,
    ):
        kernel_meta = kernel_meta or {}

        self.kernel_name = kernel_name
        self.kernel_meta = kernel_meta

        self.dynamic = kernel_meta.get("dynamic")
        self.mutated_indices = kernel_meta.get("mutated_indices") or []
        self.kernel_hash = kernel_meta.get("kernel_hash")
        self.signature = kernel_meta.get("signature")
        self.ranks = kernel_meta.get("ranks")
        self.num_outputs = kernel_meta.get("num_outputs")
        self.num_call_functions = kernel_meta.get("num_call_functions")
        self.device_index = kernel_meta.get("device_index", 0)
        self.traced_graph_hash = kernel_meta.get("traced_graph_hash")

        self.multiprocess_compile = multiprocess_compile
        self.no_more_compile = no_more_compile
        self.autotune = autotune

        self._fallback_call: Optional[Callable[..., Any]] = None
        self._nc_input_indices: Optional[List[int]] = None
        self._nc_output_indices: Optional[List[int]] = None

        self.autotuned: bool = False
        self.launchers: list[Callable[..., Any]] = []
        self.kernel_paths: list[Any] = []
        self.is_fallback_kernels: list[bool] = []

    def register_launcher(
        self,
        launcher: Callable[..., Any],
        kernel_path: Any = None,
        is_fallback_kernel: bool = False,
    ) -> None:
        """Register a compiled or fallback launcher for runtime execution."""
        self.launchers.append(launcher)
        self.kernel_paths.append(kernel_path)
        self.is_fallback_kernels.append(is_fallback_kernel)

    def get_primary_launcher_index(self) -> int:
        if not self.launchers:
            raise RuntimeError("No valid launcher")
        for idx, is_fallback_kernel in enumerate(self.is_fallback_kernels):
            if not is_fallback_kernel:
                return idx
        return 0

    def get_primary_launcher(self) -> Callable[..., Any]:
        return self.launchers[self.get_primary_launcher_index()]

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
            args_new = self.prepare_runtime_args(
                list(args),
            )
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

    def compile(self, *args, **kwargs):
        raise NotImplementedError

    def ensure_runtime_ready(self, *args, **kwargs) -> None:
        if not self.autotuned:
            if self.autotune:
                self.register_fx_fallback(self.kernel_meta)
                self.autotune_to_one_config(*args, **kwargs)
            self.autotuned = True

    def prepare_runtime_args(
        self,
        args_list: List[Any],
    ) -> List[Any]:
        args_new = ()
        for arg in args_list:
            if not torch.is_tensor(arg):
                args_new = args_new + (arg,)
                continue
            args_new = args_new + (arg, arg, 0) + arg.size() + arg.stride()
        args_list = list(args_new)
        return args_list

    def autotune_to_one_config(self, *args, **kwargs):
        return None

    def _normalize_contiguous_args(
        self, args: List[Any]
    ) -> Tuple[List[Any], Optional[List[torch.Tensor]]]:
        meta = self.kernel_meta or {}
        num_outputs = int(meta.get("num_outputs") or 0)
        num_call_functions = int(meta.get("num_call_functions") or 0)

        if num_outputs < 0 or len(args) < num_outputs:
            return args, None

        num_inputs = len(args) - num_outputs

        if self._nc_input_indices is None:
            self._nc_input_indices = []
            if num_call_functions > 0:
                input_slice = args[:num_inputs] if num_outputs > 0 else args
                for idx, arg in enumerate(input_slice):
                    if not isinstance(arg, torch.Tensor) or arg.is_contiguous():
                        continue
                    args[idx] = arg.contiguous()
                    self._nc_input_indices.append(idx)
        else:
            for idx in self._nc_input_indices:
                if idx < len(args) and isinstance(args[idx], torch.Tensor):
                    args[idx] = args[idx].contiguous()

        if num_outputs == 0:
            return args, None

        original_outputs: Optional[List[torch.Tensor]] = None
        if self._nc_output_indices is None:
            self._nc_output_indices = []
            original_outputs = []
            for j, out in enumerate(args[num_inputs:]):
                if not isinstance(out, torch.Tensor) or out.is_contiguous():
                    continue
                tmp = torch.empty(out.shape, dtype=out.dtype, device=out.device)
                out_idx = num_inputs + j
                original_outputs.append(out)
                args[out_idx] = tmp
                self._nc_output_indices.append(out_idx)
        else:
            original_outputs = []
            for out_idx in self._nc_output_indices:
                tmp = torch.empty(
                    args[out_idx].shape, dtype=args[out_idx].dtype, device=args[out_idx].device
                )
                original_outputs.append(args[out_idx])
                args[out_idx] = tmp

        return args, original_outputs

    def _copy_back_non_contiguous_outputs(
        self,
        args_list: List[Any],
        original_outputs: Optional[List[torch.Tensor]],
    ) -> None:
        if original_outputs and self._nc_output_indices:
            for orig, idx in zip(original_outputs, self._nc_output_indices):
                orig.copy_(args_list[idx])


    def _fx_graph_call_factory(self, module: torch.nn.Module, num_outputs: int) -> Callable[..., Any]:
        def module_call(*args, **kwargs):
            actual_args = args[:-num_outputs] if num_outputs > 0 else args
            actual_outputs = module.forward(*actual_args)
            for out1, out2 in zip(actual_outputs, args[-num_outputs:]):
                if isinstance(out1, torch.Tensor) and not out1.is_contiguous():
                    out1 = out1.contiguous()
                out2.data = out1.data
        return module_call

    def _load_traced_graph_model(self) -> torch.nn.Module:
        meta = self.kernel_meta or {}
        traced_graph_hash = meta.get("traced_graph_hash")
        if not traced_graph_hash:
            raise RuntimeError("traced_graph_hash missing, cannot build FX fallback")

        traced_graph_cache = meta.get("traced_graph_cache")
        if traced_graph_cache is None:
            raise RuntimeError("traced_graph_cache missing, cannot locate traced graph dump")

        base_cache = os.getenv("TORCHINDUCTOR_CACHE_DIR", "")
        dump_root = os.path.join(base_cache, traced_graph_cache)
        dump_path = os.path.join(dump_root, str(self.device_index), traced_graph_hash)

        sys.path.append(dump_path)
        try:
            module = importlib.import_module(traced_graph_hash)
        finally:
            sys.path.remove(dump_path)

        Model = getattr(module, traced_graph_hash, None)
        if Model is None:
            raise RuntimeError("Cannot find valid graph module class in traced graph dump")

        return Model()

    def register_fx_fallback(self, kernel_meta) -> None:
        model = self._load_traced_graph_model()
        num_outputs = kernel_meta.get("num_outputs", 0)
        module_call = self._fx_graph_call_factory(model, num_outputs)
        self.register_launcher(
            module_call,
            kernel_path=self.kernel_name + "_fx_fallback",
            is_fallback_kernel=True,
        )

    def run(self, *args, **kwargs):
        args_list = list(args)
        args_list, original_outputs = self._normalize_contiguous_args(args_list)
        self.ensure_runtime_ready(*args_list, **kwargs)

        launcher_idx = self.get_primary_launcher_index()
        launcher = self.launchers[launcher_idx]
        is_fallback_kernel = self.is_fallback_kernels[launcher_idx]
        if anir_config.online_acc_comp and not is_fallback_kernel:
            output = self.acc_compare_and_dump(*args_list, **kwargs)
            self._copy_back_non_contiguous_outputs(args_list, original_outputs)
            return output
        if self.dynamic and not is_fallback_kernel:
            args_list = self.prepare_runtime_args(
                args_list,
            )
        ret = launcher(*tuple(args_list), **kwargs)
        self._copy_back_non_contiguous_outputs(args_list, original_outputs)
        return ret