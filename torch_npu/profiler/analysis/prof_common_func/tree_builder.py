from queue import Queue

from ..prof_bean.node_info_bean import NodeInfoBean
from ..prof_bean.torch_op_node import TorchOpNode


class TreeBuilder:
    @classmethod
    def build_tree(cls, event_list: list) -> TorchOpNode:
        root_node = TorchOpNode(all_node_num=len(event_list))
        event_list.sort(key=lambda x: x.ts)
        last_node = root_node
        for event in event_list:
            while last_node:
                if last_node == root_node or event.ts < last_node.end_time:
                    tree_node = TorchOpNode(event, last_node)
                    last_node.add_child_node(tree_node)
                    last_node = tree_node
                    break
                last_node = last_node.parent_node
        return root_node

    @classmethod
    def find_call_node(cls, enqueue_ts: float, node_info_bean: NodeInfoBean, root_node: TorchOpNode):
        matched_child_node = root_node.match_child_node(enqueue_ts)
        if not matched_child_node:
            return
        node_queue = Queue()
        node_queue.put(matched_child_node)
        while not node_queue.empty():
            tree_node = node_queue.get()
            tree_node.update_device_total(node_info_bean)
            tree_node.update_device_range(node_info_bean)
            matched_child_node = tree_node.match_child_node(enqueue_ts)
            if matched_child_node:
                node_queue.put(matched_child_node)
            else:
                tree_node.update_device_self(node_info_bean)

    @classmethod
    def go_through_tree(cls, root_node: TorchOpNode) -> list:
        result_list = [None] * root_node.all_node_num
        node_queue = Queue()
        for child_node in root_node.child_node_list:
            node_queue.put(child_node)
        index = 0
        while not node_queue.empty():
            result_list[index] = node_queue.get()
            for child_node in result_list[index].child_node_list:
                node_queue.put(child_node)
            index += 1
        return result_list
