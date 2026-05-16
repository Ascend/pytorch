
import os
from typing import List, Dict
import torch
from torch._inductor.scheduler import BaseSchedulerNode
from .parallelism_strategy_base import ParallelStrategyBase
from .utils.schedule_node_utils import get_predecessors
from .utils.fx_pass_level import ComputeType, GroupType
from .parallelism_strategy_default import DefaultParallelStrategy
from torch._inductor.ir import Pointwise, Reduction
from ..codegen.catlass.catlass_kernel import CATLASSTemplateBuffer
from ..config import log


class CVParallelismStrategy(ParallelStrategyBase):

    def classify_node(self, node_obj):
        if node_obj is None:
            return ComputeType.UNKNOWN
        if hasattr(node_obj, "node") and isinstance(node_obj.node, CATLASSTemplateBuffer):
            return ComputeType.CUBE
        kernel_node = getattr(node_obj, 'node', None)
        if kernel_node is not None:
            return self.classify_node(kernel_node)

        snodes = getattr(node_obj, 'snodes', None)
        if snodes and len(snodes) > 0:
            # fused pointwise → vector
            return ComputeType.VECTOR

        node_class = node_obj.__class__.__name__.lower()
        
        if "multioutput" in node_class:
            inputs = getattr(node_obj, 'inputs', [])
            if inputs and len(inputs) > 0:
                return self.classify_node(inputs[0])

        python_kernel = getattr(node_obj, 'python_kernel_name', '') or ''
        cpp_kernel = getattr(node_obj, 'cpp_kernel_name', '') or ''
        op_overload = str(getattr(node_obj, 'op_overload', '')) or ''
        node_data = getattr(node_obj, 'data', None)
        data_class = ""
        if node_data:
            data_class = node_data.__class__.__name__

        combined = (
            python_kernel +
            cpp_kernel +
            op_overload +
            node_class +
            data_class
        ).lower()
        # Cube
        if any(kw in combined for kw in [
            'convolution', 'conv', 'mm', 'addmm', 'bmm', 'matmul', 'linear', 'einsum'
        ]):
            return ComputeType.CUBE
        
        if isinstance(node_data, (Pointwise, Reduction)) and hasattr(node_obj, "origins"):
            origin_ops = [str(o) for o in node_obj.origins]
            complex_ops = ['softmax', 'norm', 'layer_norm']
            if any(cop in op.lower() for op in origin_ops for cop in complex_ops):
                return ComputeType.VECTOR
            else:
                return ComputeType.MLP_VECTOR

        # MLP Vector
        if any(kw in combined for kw in [
            'add', 'relu', 'gelu', 'mul', 'bias', 'sigmoid', 'silu'
        ]):
            return ComputeType.MLP_VECTOR
        
        if 'pointwise' in data_class.lower():
            return ComputeType.VECTOR
        return ComputeType.UNKNOWN


    def extract_special_groups(self, group: List[BaseSchedulerNode], nodes: List[BaseSchedulerNode]) -> tuple[List[BaseSchedulerNode], List[BaseSchedulerNode], List[BaseSchedulerNode]]:
        if not group:
            return [], [], []
        cube_part: List[BaseSchedulerNode] = []
        vector_part: List[BaseSchedulerNode] = []
        i = 0
        n = len(group)

        while i < n:
            curr_type = group[i].compute_unit

            if curr_type == ComputeType.CUBE and not cube_part:
                while i < n:
                    if group[i].compute_unit == ComputeType.CUBE:
                        cube_part.append(group[i])
                        i += 1
                    elif i + 1 < n and group[i].compute_unit == ComputeType.MLP_VECTOR and group[i + 1].compute_unit == ComputeType.CUBE:
                        cube_part.append(group[i])
                        cube_part.append(group[i + 1])
                        i += 2
                    else:
                        break
                continue
            elif (curr_type == ComputeType.VECTOR or curr_type == ComputeType.MLP_VECTOR) and not vector_part:
                while i < n and group[i].compute_unit in (ComputeType.VECTOR, ComputeType.MLP_VECTOR, ComputeType.SCALAR):
                    vector_part.append(group[i])
                    i += 1
                continue
            i += 1
        final_cube = self._get_longest_continuous_segment(cube_part, nodes)
        final_vector = self._get_longest_continuous_segment(vector_part, nodes)
        return final_cube, final_vector


    def check_dependency(self, nodes, first_node, second_node):
        name_to_node = {n.get_name(): n for n in nodes}
        second_node_preds = get_predecessors(second_node, name_to_node)
        if first_node in second_node_preds:
            return True
        return False


    def _get_longest_continuous_segment(self, candidates: List[BaseSchedulerNode], nodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
        if not candidates:
            return []
        pos_map = {node: idx for idx, node in enumerate(nodes) if node in candidates}
        if not pos_map:
            return []
        sorted_candidates = sorted(candidates, key=lambda x: pos_map.get(x, float('inf')))
        max_segment = []
        current_segment = [sorted_candidates[0]]
        
        for i in range(1, len(sorted_candidates)):
            prev_node = sorted_candidates[i-1]
            curr_node = sorted_candidates[i]
            
            if pos_map[curr_node] == pos_map[prev_node] + 1:
                current_segment.append(curr_node)
            else:
                if len(current_segment) > len(max_segment):
                    max_segment = current_segment[:]
                current_segment = [curr_node]
        if len(current_segment) > len(max_segment):
            max_segment = current_segment

        return max_segment


    def calculate_group_len(self, group_nodes):
        group_nodes_len = 0
        for n in group_nodes:
            if isinstance(n, torch._inductor.scheduler.FusedSchedulerNode):
                snodes = getattr(n,'snodes')
                group_nodes_len += len(snodes)
            else:
                group_nodes_len += 1
        return group_nodes_len


    def assign_parallel_groups(self, nodes: List[BaseSchedulerNode]) -> Dict[str, List[BaseSchedulerNode]]:
        if not nodes:
            return {}
        for node in nodes:
            node.compute_unit = self.classify_node(node)
        default_strategy = DefaultParallelStrategy()
        groups = default_strategy.assign_parallel_groups(nodes)
        if len(groups) < 3:
            return {}
        groups_keys = list(groups.keys())
        # 计算每个汇聚节点分组结果中计算单元相同的子组
        cube0, vec0 = self.extract_special_groups(groups[groups_keys[0]], nodes)
        cube1, vec1 = self.extract_special_groups(groups[groups_keys[1]], nodes)
        # 获取最大的分组结果
        if len(cube0) >= len(cube1):
            final_cube = cube0
        else:
            final_cube = cube1

        if len(vec0) >= len(vec1):
            final_vec = vec0
        else:
            final_vec = vec1
        invalidate_all = False
        # 判断cv分组结果是否存在依赖关系
        if final_cube and final_vec:
            if self.check_dependency(nodes, final_vec[-1], final_cube[0]) or self.check_dependency(nodes, final_cube[-1], final_vec[0]):
                invalidate_all = True
        # 计算每个分组中node节点数量，如果小于最小值则不分组
        vector_group_len = self.calculate_group_len(final_vec)
        cube_group_len = self.calculate_group_len(final_cube)
        
        sub_nodes_min = os.environ.get("PARALLEL_SCHEDULER_NODES_MIN", 20) // 5
        if cube_group_len < sub_nodes_min or vector_group_len < sub_nodes_min:
            invalidate_all = True
        if invalidate_all:
            final_cube = []
            final_vec = []
        main_group = [node for node in nodes if node not in final_cube and node not in final_vec]

        final_groups: Dict[str, List[BaseSchedulerNode]] = {
            GroupType.CUBE.name: final_cube,
            GroupType.VECTOR.name: final_vec,
            GroupType.MAIN.name: main_group
        }
        log.info(f"cv parallel group len: {len(final_groups)}")
        for key, g in final_groups.items():
            names = [n.get_name() for n in g]
            log.info(f"Group {key} ({len(g)} nodes): {names}")
        return final_groups