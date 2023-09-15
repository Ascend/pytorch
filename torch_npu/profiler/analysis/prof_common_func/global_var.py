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
                cls.step_range.append([step_id, level1_node.device_start, level1_node.device_end])
        cls.torch_op_tree_node = TreeBuilder.go_through_tree(root_node)

    @classmethod
    def get_step_id_list(cls):
        if not cls.step_range:
            return [{"step_id": None, "start_ts": 0, "end_ts": float('inf'), "comm_ops": {}}]
        else:
            return [{"step_id": steps[0], "start_ts": steps[1], "end_ts": steps[2], "comm_ops": {}}
                    for steps in cls.step_range]
