import os
import re

from ..prof_bean.torch_op_bean import TorchOpBean
from ..prof_common_func.binary_decoder import BinaryDecoder
from ..prof_common_func.constant import Constant
from ..prof_common_func.file_manager import FileManager
from ..prof_common_func.file_tag import FileTag
from ..prof_common_func.path_manager import ProfilerPathManager
from ..prof_common_func.tlv_decoder import TLVDecoder
from ..prof_common_func.trace_event_manager import TraceEventManager
from ..prof_common_func.tree_builder import TreeBuilder
from ..prof_config.fwk_file_parser_config import FwkFileParserConfig


class FwkFileParser:
    def __init__(self, profiler_path: str):
        self._fwk_path = ProfilerPathManager.get_fwk_path(profiler_path)
        self._file_list = {}
        self._file_dispatch()

    def get_file_data_by_tag(self, file_tag: int) -> list:
        file_path = self._file_list.get(file_tag)
        if not file_path:
            return []
        all_bytes = FileManager.file_read_all(file_path, "rb")
        file_bean = FwkFileParserConfig.FILE_BEAN_MAP.get(file_tag, {}).get("bean")
        is_tlv = FwkFileParserConfig.FILE_BEAN_MAP.get(file_tag, {}).get("is_tlv")
        struct_size = FwkFileParserConfig.FILE_BEAN_MAP.get(file_tag, {}).get("struct_size")
        if is_tlv:
            return TLVDecoder.decode(all_bytes, file_bean, struct_size)
        else:
            return BinaryDecoder.decode(all_bytes, file_bean, struct_size)

    def get_enqueue_data(self) -> list:
        enqueue_data_list = []
        op_mark_data = self.get_file_data_by_tag(FileTag.OP_MARK)
        if not op_mark_data:
            return enqueue_data_list
        op_mark_data.sort(key=lambda x: x.time_ns)
        enqueue_start = None
        for op_mark in op_mark_data:
            if not op_mark.is_enqueue:
                continue
            if op_mark.is_enqueue_start:
                enqueue_start = op_mark
                continue
            if enqueue_start and enqueue_start.tid == op_mark.tid and enqueue_start.origin_name == op_mark.origin_name:
                op_mark.ts = enqueue_start.time_ns
                op_mark.dur = op_mark.time_ns - enqueue_start.time_ns
                enqueue_data_list.append(op_mark)
            enqueue_start = None
        return enqueue_data_list

    def get_dequeue_data(self) -> list:
        dequeue_data_list = []
        op_mark_data = self.get_file_data_by_tag(FileTag.OP_MARK)
        if not op_mark_data:
            return dequeue_data_list
        op_mark_data.sort(key=lambda x: x.time_ns)
        dequeue_start = None
        for op_mark in op_mark_data:
            if not op_mark.is_dequeue:
                continue
            if op_mark.is_dequeue_start:
                dequeue_start = op_mark
                continue
            if dequeue_start and dequeue_start.tid == op_mark.tid and dequeue_start.origin_name == op_mark.origin_name:
                op_mark.ts = dequeue_start.time_ns
                op_mark.dur = op_mark.time_ns - dequeue_start.time_ns
                dequeue_data_list.append(op_mark)
            dequeue_start = None
        return dequeue_data_list

    def get_task_queue_data(self) -> any:
        enqueue_data_list, dequeue_data_list = [], []
        op_mark_data = self.get_file_data_by_tag(FileTag.OP_MARK)
        if not op_mark_data:
            return [], []
        op_mark_data.sort(key=lambda x: x.time_ns)
        enqueue_start = None
        dequeue_start = None
        for op_mark in op_mark_data:
            if op_mark.is_enqueue_start:
                enqueue_start = op_mark
                continue
            if op_mark.is_dequeue_start:
                dequeue_start = op_mark
                continue
            if op_mark.is_enqueue_end and enqueue_start:
                if enqueue_start.tid == op_mark.tid and enqueue_start.origin_name == op_mark.origin_name:
                    op_mark.ts = enqueue_start.time_ns
                    op_mark.dur = op_mark.time_ns - enqueue_start.time_ns
                    enqueue_data_list.append(op_mark)
                    enqueue_start = None
                continue
            if op_mark.is_dequeue_end and dequeue_start:
                if dequeue_start.tid == op_mark.tid and dequeue_start.origin_name == op_mark.origin_name:
                    op_mark.ts = dequeue_start.time_ns
                    op_mark.dur = op_mark.time_ns - dequeue_start.time_ns
                    dequeue_data_list.append(op_mark)
                    dequeue_start = None
        return enqueue_data_list, dequeue_data_list

    def get_torch_op_tree_node(self, only_fwk: bool = False) -> list:
        torch_op_list = self.get_file_data_by_tag(FileTag.TORCH_OP)
        if not torch_op_list:
            return []
        enqueue_data_list = []
        if not only_fwk:
            enqueue_data_list = self.get_enqueue_data()
        result_data = TreeBuilder.build_tree(torch_op_list, enqueue_data_list)
        return result_data

    def get_fwk_trace_data(self):
        torch_op_data = self.get_file_data_by_tag(FileTag.TORCH_OP)
        if not torch_op_data:
            return []
        enqueue_data_list, dequeue_data_list = self.get_task_queue_data()
        pid = torch_op_data[0].pid
        tid_dict = {}
        fwk_x_event_list = [None] * (
                len(torch_op_data) + len(enqueue_data_list) * 2 + len(dequeue_data_list) * 2)
        index = 0
        fwd_dict = {}
        for torch_op in torch_op_data:
            self.filter_fwd_bwd_event(fwd_dict, torch_op)
            tid_dict[torch_op.tid] = False
            fwk_x_event_list[index] = TraceEventManager.create_x_event(torch_op, "cpu_op")
            index += 1
        for enqueue_data in enqueue_data_list:
            tid_dict[enqueue_data.tid] = False
            fwk_x_event_list[index] = TraceEventManager.create_x_event(enqueue_data, "enqueue")
            index += 1
            fwk_x_event_list[index] = TraceEventManager.create_task_queue_flow(Constant.FLOW_START_PH, enqueue_data)
            index += 1
        for dequeue_data in dequeue_data_list:
            tid_dict[dequeue_data.tid] = True
            fwk_x_event_list[index] = TraceEventManager.create_x_event(dequeue_data, "dequeue")
            index += 1
            fwk_x_event_list[index] = TraceEventManager.create_task_queue_flow(Constant.FLOW_END_PH, dequeue_data)
            index += 1
        other_event_list = TraceEventManager.create_m_event(pid, tid_dict)
        other_event_list.extend(TraceEventManager.create_fwd_flow(fwd_dict))
        fwk_x_event_list.extend(other_event_list)
        return fwk_x_event_list

    @classmethod
    def filter_fwd_bwd_event(cls, fwd_dict: dict, torch_op: TorchOpBean):
        seq_num = torch_op.args.get("Sequence number", -1)
        if seq_num < 0:
            return
        fwd_event = fwd_dict.get(seq_num, {})
        mode = "start" if torch_op.args.get("Fwd thread id") == 0 else "end"
        if fwd_event.get(mode, {}).get("ts", -float('inf')) < torch_op.ts:
            node = {mode: {'pid': torch_op.pid, 'tid': torch_op.tid, 'ts': torch_op.ts}}
            fwd_dict.setdefault(seq_num, {}).update(node)

    def has_task_queue_data(self):
        return bool(self._file_list.get(FileTag.OP_MARK))

    def _file_dispatch(self):
        for file_name in os.listdir(os.path.realpath(self._fwk_path)):
            file_path = os.path.join(self._fwk_path, file_name)
            if not os.path.isfile(file_path):
                continue
            for file_tag, pattern in FwkFileParserConfig.FILE_DISPATCH_MAP.items():
                if re.match(pattern, file_name):
                    self._file_list.setdefault(file_tag, file_path)
