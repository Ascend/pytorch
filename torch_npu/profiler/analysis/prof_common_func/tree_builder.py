# Copyright (c) 2023, Huawei Technologies.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from queue import Queue

from ..prof_bean.op_mark_bean import OpMarkBean
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
    def update_tree_node_info(cls, info_data: any, root_node: TorchOpNode):
        if isinstance(info_data, OpMarkBean):
            ts = info_data.ts
            corr_id = info_data.corr_id
        else:
            ts = info_data
            corr_id = info_data
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

    @classmethod
    def go_through_tree(cls, root_node: TorchOpNode) -> list:
        if not root_node.all_node_num:
            return []
        result_list = [None] * (root_node.all_node_num + 1)
        result_list[0] = root_node
        node_queue = Queue()
        for child_node in root_node.child_node_list:
            node_queue.put(child_node)
        index = 1
        while not node_queue.empty():
            result_list[index] = node_queue.get()
            for child_node in result_list[index].child_node_list:
                node_queue.put(child_node)
            index += 1
        return result_list
