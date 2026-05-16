
from typing import List, Dict, Set
from collections import defaultdict
import torch
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
        # 遍历所有节点，构建每个节点的前驱节点后后继节点以及入度
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
        # 找到汇聚节点(入度>1)
        convergences = [node for node in nodes if indegree[node.get_name()] > 1]
        if not convergences:
            return []
        convergences_sorted = sorted(convergences, key=lambda x: node_to_idx[x])
        split_merge = None
        # 倒排找到最靠后的且刚好有两个前驱的merge节点
        for conv in reversed(convergences_sorted):
            preds_count = len(predecessors.get(conv.get_name(), []))
            if preds_count == 2:
                split_merge = conv
                break
        if split_merge is None:
            return []
        # 找到merge前的两个分支的起点
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
        # 找到每个分支起点的所有祖先节点
        for pre in sorted_preds:
            anc_sets.append(get_ancestors_node(pre))
        # 处理两个分支中的共享节点，从共享节点进行截取
        anc_sets = make_disjoint(anc_sets)
        # 最终分支内节点按nodes原定顺序排序
        for anc_set in anc_sets:
            sorted_group = sorted(anc_set, key=lambda x: node_to_idx[x])
            groups.append(sorted_group)

        temp_groups = []
        final_groups = dict()
        # 处理分组后节点的连续性(原始nodes中的连续)，去除中间断开的节点
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
        # 检查两个分组中是否存在依赖，找到依赖点
        for i, n in enumerate(reversed(group_2)):
            pre_nodes = predecessors.get(n.get_name(), [])
            first_group_index = find_first_overlap(pre_nodes, group_1)
            if first_group_index:
                second_group_end_flag = len(group_2) - 1 - i
                first_group_end_flag = first_group_index + 1
        # 根据依赖点截断调整分组结果
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

        log.info(f"default parallel group len: {len(final_groups)}")
        for key, g in final_groups.items():
            names = [n.get_name() for n in g]
            log.info(f"Group {key} ({len(g)} nodes): {names}")

        return final_groups