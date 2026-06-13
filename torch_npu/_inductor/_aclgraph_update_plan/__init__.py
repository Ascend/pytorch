from torch_npu._inductor._aclgraph_update_plan.codegen import (
    append_inductor_aclgraph_update_plan_for_codegen_node,
    emit_inductor_aclgraph_update_plan_for_wrapper,
)
from torch_npu.npu._aclgraph_update_plan import ACLGRAPH_UPDATE_PLAN_GLOBAL


__all__ = [
    "ACLGRAPH_UPDATE_PLAN_GLOBAL",
    "append_inductor_aclgraph_update_plan_for_codegen_node",
    "emit_inductor_aclgraph_update_plan_for_wrapper",
]
