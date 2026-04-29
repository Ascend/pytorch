import os
from torch._inductor.scheduler import BaseSchedulerNode
from typing import Dict, List, Set


def is_multi_stream():
    return os.environ.get("PARALLEL_SCHEDULER_OPTIMIZAR", "false").lower() == "true"

def find_first_overlap(pre_nodes, first_group_nodes):
    for i, item in enumerate(reversed(first_group_nodes)):
        if item in pre_nodes:
            return len(first_group_nodes) - 1 - i
    return None
    

def make_disjoint(anc_sets):
    result = []
    seen = set()
    for s in anc_sets:
        cleaned = s - seen
        result.append(cleaned)
        seen |= s
    return result


def get_predecessors(
    node: BaseSchedulerNode,
    name_to_node: Dict[str, BaseSchedulerNode]
) -> Set[BaseSchedulerNode]:
    preds = set()

    if hasattr(node, 'mpi_node') and node.mpi_node is not None:
        for pred_node in node.mpi_node.pred_nodes:
            if isinstance(pred_node, BaseSchedulerNode):
                preds.add(pred_node)

    if preds:
        return preds

    if hasattr(node, 'ancestors') and node.ancestors:
        for name in node.ancestors:
            if name in name_to_node:
                preds.add(name_to_node[name])

    if hasattr(node, 'read_writes') and node.read_writes.reads:
        for dep in node.read_writes.reads:
            if hasattr(dep, 'name') and dep.name in name_to_node:
                preds.add(name_to_node[dep.name])

    if hasattr(node, 'unmet_dependencies'):
        for dep in node.unmet_dependencies:
            if hasattr(dep, 'name') and dep.name in name_to_node:
                preds.add(name_to_node[dep.name])
    return preds


def get_successors_names(node: BaseSchedulerNode) -> List[str]:
    succ_nodes = set()
    if hasattr(node, 'mpi_node') and node.mpi_node is not None:
        for succ in node.mpi_node.succ_nodes:
            if isinstance(succ, BaseSchedulerNode):
                succ_nodes.add(succ)
    return succ_nodes
