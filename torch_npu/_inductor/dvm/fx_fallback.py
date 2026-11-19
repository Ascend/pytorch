"""FX graph fallback loading and execution, with PyTorch async compile support."""

import importlib
import os
import sys
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor.async_compile import AsyncCompile, _compile_start


class FxGraphFallback:
    def __init__(self, kernel_name="", kernel_meta=None):
        self.kernel_name = kernel_name
        self.kernel_meta = kernel_meta or {}
        self.device_index = self.kernel_meta.get("device_index", 0)
        self._nc_input_indices = None
        self._nc_output_indices = None
        self._launcher: Optional[Callable[..., Any]] = None

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
                if out1.device == out2.device and out1.dtype == out2.dtype:
                    out2.set_(out1)
                else:
                    # clone_for_accuracy upcasts bfloat16 outputs to float32, while the model
                    # forward still produces bfloat16. set_ fails due to dtype mismatch,
                    # fall back to copy_ for cross-dtype copy.
                    out2.copy_(out1)
        return module_call

    def _load_traced_graph_model(self) -> torch.nn.Module:
        meta = self.kernel_meta or {}
        if "npu_fx_asset_id" in meta:
            from torch_npu._inductor.dvm.fx_asset import load_fx_asset

            # The graph-owned asset is persisted by the wrapper and carries the
            # real constants; a to_folder dump only rebuilds empty tensors.
            return load_fx_asset(meta["npu_fx_asset_id"])
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
        from torch_npu.dynamo._package_compat.format import PackageCompatibilityError

        try:
            model = self._load_traced_graph_model()
        except PackageCompatibilityError as error:
            # The cached FX entry outlived its assets (partial cache cleanup or
            # corruption); a fresh compile rebuilds both.
            raise PackageCompatibilityError(
                f"{error} (kernel {self.kernel_name}); the cached FX entry "
                "outlived its assets. Remove the TORCHINDUCTOR_CACHE_DIR "
                "contents and run again: a fresh compile rebuilds both."
            ) from error
        num_outputs = kernel_meta.get("num_outputs", 0)
        self._launcher = self._fx_graph_call_factory(model, num_outputs)

    def run(self, *args, **kwargs):
        args_list, original_outputs = self._normalize_contiguous_args(list(args))
        launcher = self._launcher
        if launcher is None:
            raise RuntimeError("No valid launcher")
        kwargs = {k: v for k, v in kwargs.items() if k != "stream"}
        ret = launcher(*tuple(args_list), **kwargs)
        self._copy_back_non_contiguous_outputs(args_list, original_outputs)
        return ret


def _load_fx_graph(kernel_name: str, kernel_meta: dict[str, Any]) -> FxGraphFallback:
    kernel = FxGraphFallback(kernel_name, kernel_meta=kernel_meta)
    kernel.register_fx_fallback(kernel_meta)
    os.makedirs(
        os.path.join(
            os.getenv("TORCHINDUCTOR_CACHE_DIR", ""),
            kernel_meta.get("traced_graph_cache"),
            str(kernel_meta.get("device_index")),
            kernel_meta.get("traced_graph_hash"),
            "keep",
        ),
        exist_ok=True,
    )
    return kernel


class CustomAsyncCompile(AsyncCompile):
    def import_fx(self, module_name: str, kernel_meta: dict[str, Any]) -> FxGraphFallback:
        device_interface = get_interface_for_device(kernel_meta.get("device_str"))
        device_interface.current_device()
        _compile_start()
        return _load_fx_graph(module_name, kernel_meta=kernel_meta)
