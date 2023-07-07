from .tree_builder import TreeBuilder
from ..prof_parse.fwk_cann_relation_parser import FwkCANNRelationParser


class GlobalVar:
    torch_op_tree_node = []
    step_range = []

    @classmethod
    def init(cls, profiler_path: str):
        root_node = FwkCANNRelationParser(profiler_path).build_torch_op_tree()
        if not root_node.child_node_list:
            return
        for level1_node in root_node.child_node_list:
            if level1_node.is_profiler_step():
                step_id = level1_node.event.name.split("#")[-1]
                cls.step_range.append([step_id, level1_node.first_kernel_ts, level1_node.end_kernel_ts])
        cls.torch_op_tree_node = TreeBuilder.go_through_tree(root_node)
