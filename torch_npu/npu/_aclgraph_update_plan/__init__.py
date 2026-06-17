from torch_npu.npu._aclgraph_update_plan.resolver import (
    ACLGRAPH_UPDATE_PLAN_GLOBAL,
    build_cpu_update_input_for_graph,
    resolve_aclgraph_update_plan,
    update_aclgraph_records_for_graph,
    validate_aclgraph_update_plan,
    validate_aclgraph_update_plan_for_graph,
)


__all__ = [
    "ACLGRAPH_UPDATE_PLAN_GLOBAL",
    "build_cpu_update_input_for_graph",
    "resolve_aclgraph_update_plan",
    "update_aclgraph_records_for_graph",
    "validate_aclgraph_update_plan",
    "validate_aclgraph_update_plan_for_graph",
]
