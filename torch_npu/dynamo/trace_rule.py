import torch
from torch._dynamo.variables import TorchInGraphFunctionVariable
from torch._dynamo.trace_rules import manual_torch_name_rule_map, SkipFunctionVariable
import torch_npu

__all__ = []

torch_c_binding_in_graph_functions_npu = dict.fromkeys(
    [
        "torch.npu.current_stream",
        "torch.npu.default_stream",
        "torch.npu.stream",
        "torch.npu.set_stream",
    ],
    TorchInGraphFunctionVariable,
)


def _patch_npu_trace_rules():
    torch._dynamo.trace_rules.clear_lru_cache()
    torch._dynamo.trace_rules.torch_name_rule_map.append(torch_c_binding_in_graph_functions_npu)
    manual_torch_name_rule_map["torch_npu.npu.utils.synchronize"] = SkipFunctionVariable
