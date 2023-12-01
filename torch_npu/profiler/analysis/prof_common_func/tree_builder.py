from queue import Queue

from ..prof_bean.op_mark_bean import OpMarkBean
from ..prof_bean.torch_op_node import TorchOpNode


class TreeBuilder:
    @classmethod
    def build_tree(cls, event_list: list, enqueue_list: list) -> TorchOpNode:
        all_node_list = [None] * (len(event_list) + 1)
        event_list.extend(enqueue_list)
        event_list.sort(key=lambda x: x.ts)
        root_node = TorchOpNode()
        last_node = root_node
        index = 0
        all_node_list[index] = root_node
        for event in event_list:
            while last_node:
                if last_node != root_node and event.ts > last_node.end_time:
                    last_node = last_node.parent_node
                    continue
                if event.is_torch_op:
                    tree_node = TorchOpNode(event, last_node)
                    last_node.add_child_node(tree_node)
                    last_node = tree_node
                    index += 1
                    all_node_list[index] = tree_node
                else:
                    last_node.update_corr_id(event.corr_id)
                break
        return all_node_list

    @classmethod
    def update_tree_node_info(cls, acl_ts: int, root_node: TorchOpNode):
        ts = acl_ts
        corr_id = acl_ts
        matched_child_node = root_node.match_child_node(ts)
        if not matched_child_node:
            return
        node_queue = Queue()
        node_queue.put(matched_child_node)
        while not node_queue.empty():
            tree_node = node_queue.get()
            tree_node.update_corr_id_total(corr_id)
            matched_child_node = tree_node.match_child_node(ts)
            if matched_child_node:
                node_queue.put(matched_child_node)
            else:
                tree_node.update_corr_id_self(corr_id)

    @classmethod
    def match_self_torch_op(cls, ts: int, root_node: TorchOpNode) -> any:
        matched_child_node = root_node.match_child_node(ts)
        if not matched_child_node:
            return None
        node_queue = Queue()
        node_queue.put(matched_child_node)
        while not node_queue.empty():
            tree_node = node_queue.get()
            matched_child_node = tree_node.match_child_node(ts)
            if matched_child_node:
                node_queue.put(matched_child_node)
            else:
                return tree_node
