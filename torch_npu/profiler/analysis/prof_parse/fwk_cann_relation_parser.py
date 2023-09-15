from ..prof_bean.node_info_bean import NodeInfoBean
from ..prof_common_func.file_tag import FileTag
from ..prof_common_func.tree_builder import TreeBuilder, TorchOpNode
from ..prof_parse.cann_file_parser import CANNFileParser
from ..prof_parse.fwk_file_parser import FwkFileParser


class FwkCANNRelationParser:
    def __init__(self, profiler_path: str):
        self._profiler_path = profiler_path

    def build_torch_op_tree(self) -> TorchOpNode:
        fwk_parser = FwkFileParser(self._profiler_path)
        torch_op_list = fwk_parser.get_file_data_by_tag(FileTag.TORCH_OP)
        if not torch_op_list:
            return TorchOpNode()
        root_node = TreeBuilder.build_tree(torch_op_list)

        acl_to_npu_dict = CANNFileParser(self._profiler_path).get_acl_to_npu_data()
        enqueue_data_list, dequeue_data_list = fwk_parser.get_task_queue_data()
        if not acl_to_npu_dict:
            return root_node
        acl_start_time_list = sorted(list(acl_to_npu_dict.keys()))
        if not enqueue_data_list and not dequeue_data_list:
            for acl_start_time in acl_start_time_list:
                kernel_list = acl_to_npu_dict.get(acl_start_time, [])
                if not kernel_list:
                    continue
                TreeBuilder.find_call_node(acl_start_time, NodeInfoBean(kernel_list), root_node)
            return root_node

        corr_id_dict = {}
        index = 0
        for acl_start_time in acl_start_time_list:
            while index < len(dequeue_data_list):
                if dequeue_data_list[index].ts > acl_start_time:
                    break
                if acl_start_time <= dequeue_data_list[index].ts + dequeue_data_list[index].dur:
                    corr_id_dict.setdefault(dequeue_data_list[index].corr_id, []).append(acl_start_time)
                    break
                index += 1

        for enqueue_data in enqueue_data_list:
            acl_start_time_list = corr_id_dict.get(enqueue_data.corr_id, [])
            kernel_list = []
            for acl_start_time in acl_start_time_list:
                kernel_list.extend(acl_to_npu_dict.get(acl_start_time, []))
            if not kernel_list:
                continue
            TreeBuilder.find_call_node(enqueue_data.ts, NodeInfoBean(kernel_list), root_node)
        return root_node
