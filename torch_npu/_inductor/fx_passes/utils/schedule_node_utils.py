import os

from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from ...codegen.cpp_wrapper import CppWrapperNpu


def is_multi_stream():
    wrapper = V.graph.wrapper_code
    is_cpp_wrapper = isinstance(wrapper, (CppWrapperNpu, CppWrapperCpu, CppWrapperGpu))
    env_parallel = (
        os.environ.get("ENABLE_PARALLEL_SCHEDULER", "false").lower() == "true"
    )
    return env_parallel and not is_cpp_wrapper


def find_first_overlap(pre_nodes, first_group_nodes):
    for i, item in enumerate(reversed(first_group_nodes)):
        if item in pre_nodes:
            return len(first_group_nodes) - 1 - i
    return None


def make_disjoint(anc_sets):
    result = []
    seen = OrderedSet()
    for s in anc_sets:
        cleaned = s - seen
        result.append(cleaned)
        seen |= s
    return result


def get_predecessors(
    node: BaseSchedulerNode, name_to_node: dict[str, BaseSchedulerNode]
) -> OrderedSet[BaseSchedulerNode]:
    preds = OrderedSet()

    if hasattr(node, "mpi_node") and node.mpi_node is not None:
        for pred_node in node.mpi_node.pred_nodes:
            if isinstance(pred_node, BaseSchedulerNode):
                preds.add(pred_node)

    if preds:
        return preds

    if hasattr(node, "ancestors") and node.ancestors:
        for name in node.ancestors:
            if name in name_to_node:
                preds.add(name_to_node[name])

    if hasattr(node, "read_writes") and node.read_writes.reads:
        for dep in node.read_writes.reads:
            if hasattr(dep, "name") and dep.name in name_to_node:
                preds.add(name_to_node[dep.name])

    if hasattr(node, "unmet_dependencies"):
        for dep in node.unmet_dependencies:
            if hasattr(dep, "name") and dep.name in name_to_node:
                preds.add(name_to_node[dep.name])
    return preds


def get_successors_names(node: BaseSchedulerNode) -> list[str]:
    succ_nodes = OrderedSet()
    if hasattr(node, "mpi_node") and node.mpi_node is not None:
        for succ in node.mpi_node.succ_nodes:
            if isinstance(succ, BaseSchedulerNode):
                succ_nodes.add(succ)
    return succ_nodes
