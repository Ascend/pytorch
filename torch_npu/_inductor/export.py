import torch
from collections.abc import Callable
import torch.utils._pytree as pytree
from torch.export._tree_utils import reorder_kwargs

def patch_aot_load():
    origin_aot_load = torch._export.aot_load

    def aot_load_npu(so_path: str, device: str) -> Callable:

        if device == "npu" or device.startswith("npu:"):
            runner = torch._C._aoti.AOTIModelContainerRunnerNpu(so_path, 1, device)
        else:
            return origin_aot_load(so_path, device)

        def optimized(*args, **kwargs):
            call_spec = runner.get_call_spec()
            in_spec = pytree.treespec_loads(call_spec[0])
            out_spec = pytree.treespec_loads(call_spec[1])
            flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, in_spec)))[0]
            flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
            flat_outputs = runner.run(flat_inputs)
            return pytree.tree_unflatten(flat_outputs, out_spec)

        return optimized

    torch._export.aot_load = aot_load_npu
