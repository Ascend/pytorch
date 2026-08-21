
from typing import List, Dict, Set
from collections import defaultdict
from torch._inductor.scheduler import BaseSchedulerNode
from .parallelism_strategy_base import ParallelStrategyBase
from .utils.schedule_node_utils import make_disjoint, find_first_overlap, get_predecessors, get_successors_names
from .utils.fx_pass_level import GroupType
from ..config import log


class DefaultParallelStrategy(ParallelStrategyBase):

    def assign_parallel_groups(self, nodes: List[BaseSchedulerNode]) -> Dict[str, List[BaseSchedulerNode]]:
        if not nodes:
            return {}

        name_to_node = {n.get_name(): n for n in nodes}
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        predecessors: Dict[str, Set] = defaultdict(set)
        successors: Dict[str, Set] = defaultdict(set)
        indegree: Dict[str, int] = {n.get_name(): 0 for n in nodes}
        # walk all nodes, building each node's predecessors, successors and in-degree
        for node in nodes:
            my_name = node.get_name()
            preds = get_predecessors(node, name_to_node)
            successors_nodes = get_successors_names(node)
            successors[my_name].update(successors_nodes)
            for pred in preds:
                pred_name = pred.get_name()
                predecessors[my_name].add(pred)
                successors[pred_name].add(node)
                indegree[node.get_name()] += 1
        # find merge nodes (in-degree > 1)
        convergences = [node for node in nodes if indegree[node.get_name()] > 1]
        if not convergences:
            return []
        convergences_sorted = sorted(convergences, key=lambda x: node_to_idx[x])
        split_merge = None
        # scan backwards for the last merge node that has exactly two predecessors
        for conv in reversed(convergences_sorted):
            preds_count = len(predecessors.get(conv.get_name(), []))
            if preds_count == 2:
                split_merge = conv
                break
        if split_merge is None:
            return []
        # find the start of the two branches ahead of the merge
        preds = predecessors.get(split_merge.get_name(), [])
        sorted_preds = sorted(preds, key=lambda x: node_to_idx.get(x, -1))
        groups = []
        assigned = set()

        def get_ancestors_node(start):
            ancestors = set()
            stack = [start]
            while stack:
                curr = stack.pop()
                if curr in ancestors:
                    continue
                ancestors.add(curr)
                for pre in predecessors.get(curr.get_name(), []):
                    stack.append(pre)
            return ancestors
        anc_sets = []
        # collect all ancestors of each branch start
        for pre in sorted_preds:
            anc_sets.append(get_ancestors_node(pre))
        # handle nodes shared by the two branches, cutting at the shared node
        anc_sets = make_disjoint(anc_sets)
        # order the nodes inside each branch by their original order in nodes
        for anc_set in anc_sets:
            sorted_group = sorted(anc_set, key=lambda x: node_to_idx[x])
            groups.append(sorted_group)

        temp_groups = []
        final_groups = dict()
        # keep each group contiguous (in the original nodes order), dropping nodes past a break
        for group in groups:
            prefix_len = 1
            for i in range(1, len(group)):
                if node_to_idx[group[i]] == node_to_idx[group[i - 1]] + 1:
                    prefix_len += 1
                else:
                    break
            trimmed = group[:prefix_len]
            temp_groups.append(trimmed)
        group_1 = temp_groups[0]
        group_2 = temp_groups[1]
        first_group_end_flag = len(group_1)
        second_group_end_flag = len(group_2)
        # check whether the two groups have a dependency and locate the dependency point
        for i, n in enumerate(reversed(group_2)):
            pre_nodes = predecessors.get(n.get_name(), [])
            first_group_index = find_first_overlap(pre_nodes, group_1)
            if first_group_index:
                second_group_end_flag = len(group_2) - 1 - i
                first_group_end_flag = first_group_index + 1
        # truncate the groups at the dependency point
        first_group = group_1[:first_group_end_flag]
        assigned.update(first_group)
        second_group = group_2[:second_group_end_flag]
        assigned.update(second_group)
        if len(first_group) > 0:
            final_groups[GroupType.MIX_01.name] = first_group
        if len(second_group) > 0:
            final_groups[GroupType.MIX_02.name] = second_group

        main_group = [node for node in nodes if node not in assigned]
        final_groups[GroupType.MAIN.name] = main_group

        log.info("default parallel group len: %s", len(final_groups))
        for key, g in final_groups.items():
            names = [n.get_name() for n in g]
            log.info("Group %s (%s nodes): %s", key, len(g), names)

        return final_groups
